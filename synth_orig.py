import argparse, os
import numpy as np
import torch
import torch.utils.data
from copy import deepcopy
from tqdm.auto import tqdm
from scipy.io import wavfile
import warnings

from datain import DataSet


def load_model(basename, device='cpu'):
    basename = 'weights/' + basename
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = torch.load(basename + '.model.pt', map_location=device)
    return model


def synthesize(frames, filename, stride=2048, sr=16000, ymax=0.98):
    # Generate stream
    y = torch.zeros((len(frames) - 1) * stride + len(frames[0]))
    for i, x in enumerate(frames):
        y[i * stride:i * stride + len(x)] += x
    # To numpy & deemph
    y = y.numpy().astype(np.float32)
    # Normalize
    y -= np.mean(y)
    mx = np.max(np.abs(y))
    if mx > 0:
        y *= ymax / mx
    else:
        y = np.clip(y, -ymax, ymax)
    # To 16 bit & save
    wavfile.write(filename, sr, np.array(y * 32767, dtype=np.int16))
    return y


# Arguments
parser = argparse.ArgumentParser(description='Audio synthesis script')
parser.add_argument('--seed_input', default=0, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--seed', default=0, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--device', default='cuda', type=str, required=False, help='(default=%(default)s)')
# Data
parser.add_argument('--trim', default=-1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--base_fn_model', default='', type=str, required=True, help='(default=%(default)s)')
parser.add_argument('--path_out', default='../res/', type=str, required=True, help='(default=%(default)s)')
parser.add_argument('--split', default='test', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--force_source_file', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--force_source_speaker', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--force_target_speaker', default='', type=str, required=False, help='(default=%(default)s)')
# Conversion
parser.add_argument('--fn_list', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--sbatch', default=256, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--convert', action='store_true')
parser.add_argument('--zavg', action='store_true', required=False, help='(default=%(default)s)')
parser.add_argument('--alpha', default=3, type=float, required=False, help='(default=%(default)f)')
# Synthesis
parser.add_argument('--lchunk', default=-1, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--stride', default=-1, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--synth_nonorm', action='store_true')
parser.add_argument('--maxfiles', default=10000000, type=int, required=False, help='(default=%(default)d)')

# Process arguments
args = parser.parse_args()
if args.trim <= 0:
    args.trim = None
if args.force_source_file == '':
    args.force_source_file = None
if args.force_source_speaker == '':
    args.force_source_speaker = None
if args.force_target_speaker == '':
    args.force_target_speaker = None
if args.fn_list == '':
    args.fn_list = 'list_seed' + str(args.seed_input) + '_' + args.split + '.tsv'

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)

########################################################################################################################

# Load model, pars, & check

print('Load stuff')
model = load_model(args.base_fn_model)
model = model.to(args.device)
window = torch.hann_window(4096).view(1, -1)

# Data
print('Load metadata')
dataset = DataSet('data', 4096, 4096, sampling_rate=16000, split='train+valid',
                  seed=0, do_audio_load=False)
speakers = deepcopy(dataset.speakers)
lspeakers = list(speakers.keys())

# Input data
print('Load', args.split, 'audio')
dataset = DataSet('data', 4096, 4096 // 2, sampling_rate=16000, split=args.split,
                  trim=args.trim,
                  select_speaker=args.force_source_speaker, select_file=args.force_source_file,
                  seed=0)
loader = torch.utils.data.DataLoader(dataset, batch_size=args.sbatch, shuffle=False, num_workers=0)

# Get transformation list
print('Transformation list')
np.random.seed(args.seed)
target_speaker = lspeakers[np.random.randint(len(lspeakers))]
if args.force_target_speaker is not None: target_speaker = args.force_target_speaker
fnlist = []
itrafos = []
nfiles = 0
for x, info in loader:
    isource, itarget = [], []
    for n in range(len(x)):

        # Get source and target speakers
        i, j, last, ispk, iut = info[n]
        source_speaker, _ = dataset.filename_split(dataset.filenames[i])
        isource.append(speakers[source_speaker])
        itarget.append(speakers[target_speaker])
        if last == 1 and nfiles < args.maxfiles:

            # Get filename
            fn = dataset.filenames[i][:-len('.pt')]
            fnlist.append([fn, source_speaker, target_speaker])

            # Restart
            target_speaker = lspeakers[np.random.randint(len(lspeakers))]
            if args.force_target_speaker is not None: target_speaker = args.force_target_speaker
            nfiles += 1

    isource, itarget = torch.LongTensor(isource), torch.LongTensor(itarget)
    itrafos.append([isource, itarget])
    if nfiles >= args.maxfiles:
        break

# Write transformation list
flist = open(os.path.join(args.path_out, args.fn_list), 'w')
for fields in fnlist:
    flist.write('\t'.join(fields) + '\n')
flist.close()

########################################################################################################################

# Prepare model
model.precalc_matrices('on')
model.eval()

# Synthesis loop
print('Synth...')
audio, nfiles, t_conv, t_synth, t_audio = [], 0, 0, 0, 0
with torch.no_grad():
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for k, (x, info) in pbar:
        if k >= len(itrafos): break
        isource, itarget = itrafos[k]

        # Forward & reverse
        x = x.to(args.device)
        isource = isource.to(args.device)
        itarget = itarget.to(args.device)
        z = model.forward(x, isource)[0]
        x = model.reverse(z, itarget)
        x = x.cpu()
        x *= window

        for n in range(len(x)):
            audio.append(x[n])
            i, j, last, ispk, iut = info[n]

            if last == 1:
                fn, source_speaker, target_speaker = fnlist[nfiles]
                _, fn = os.path.split(fn)
                # Synthesize
                synthesize(audio, filename=f"{args.path_out}/{fn}_to_{target_speaker}.wav")
                # Reset
                audio = []
                nfiles += 1
