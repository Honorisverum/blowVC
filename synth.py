import sys, os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy.io import wavfile
from tqdm.auto import tqdm
from torch.backends import cudnn
import numba

from data import DatasetVC
from model import Model


def load_model(basename):
    basename = 'weights/' + basename
    state = torch.load(basename + '.model.pt', map_location='cpu')
    model = Model(**state['model_params'])
    model.load_state_dict(state['state_dict'], strict=True)
    return model


def init_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)


def synthesize(frames, filename, stride, sr=16000, deemph=0, ymax=0.98, normalize=False):
    # Generate stream
    y = torch.zeros((len(frames) - 1) * stride + len(frames[0]))
    for i, x in enumerate(frames):
        y[i * stride:i * stride + len(x)] += x
    # To numpy & deemph
    y = y.numpy().astype(np.float32)
    if deemph > 0:
        y = deemphasis(y, alpha=deemph)
    # Normalize
    if normalize:
        y -= np.mean(y)
        mx = np.max(np.abs(y))
        if mx > 0:
            y *= ymax / mx
    else:
        y = np.clip(y, -ymax, ymax)
    # To 16 bit & save
    wavfile.write(filename, sr, np.array(y * 32767, dtype=np.int16))
    return y


@numba.jit(nopython=True, cache=True)
def deemphasis(x, alpha=0.2):
    assert 0 <= alpha <= 1
    if alpha == 0 or alpha == 1:
        return x
    y = x.copy()
    for n in range(1, len(x)):
        y[n] = x[n] + alpha * y[n - 1]
    return y


PATH_DATA = 'data'
TRIM = None
SR = 16000
SBATCH = 256
FORCE_TARGET_SPEAKER = 'BillGates'
SEED = 0
SPLIT = 'valid'
FN_LIST = 'list_seed' + '0' + '_' + SPLIT + '.tsv'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LCHUNK = 4096
STRIDE = LCHUNK // 2
PATH_OUT = 'synth'

BASE_FN_MODEL = 'blow'
init_seed(SEED)

#################################################################################################################


print('Load', SPLIT, 'audio...')
dataset = DatasetVC(PATH_DATA, LCHUNK, STRIDE, sampling_rate=SR, split=SPLIT, trim=TRIM, seed=SEED)
loader = DataLoader(dataset, batch_size=SBATCH, shuffle=False, num_workers=0)
speakers = deepcopy(dataset.speakers)

print('Load model...')
model = load_model(BASE_FN_MODEL)
model = model.to(DEVICE)
window = torch.hann_window(LCHUNK).view(1, -1)

print('Transformation list...')
np.random.seed(SEED)
target_speaker = FORCE_TARGET_SPEAKER
fnlist, itrafos, nfiles = [], [], 0

for x, info in loader:

    isource, itarget = [], []
    for n in range(len(x)):

        # Get source and target speakers
        i, j, last, ispk, iut = info[n]
        source_speaker, _ = dataset.filename_split(dataset.filenames[i])
        isource.append(speakers[source_speaker])
        itarget.append(speakers[target_speaker])

        if last == 1:
            # Get filename
            fn = dataset.filenames[i][:-len('.pt')]
            fnlist.append([fn, source_speaker, target_speaker])
            nfiles += 1

    isource, itarget = torch.LongTensor(isource), torch.LongTensor(itarget)
    itrafos.append([isource, itarget])

# Prepare model
model.precalc_matrices('on')
model.eval()

print('Synth...')
audio, nfiles, t_conv, t_synth, t_audio = [], 0, 0, 0, 0
with torch.no_grad():
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for k, (x, info) in pbar:
        if k >= len(itrafos): break
        isource, itarget = itrafos[k]

        # Forward & reverse
        x = x.to(DEVICE)
        isource = isource.to(DEVICE)
        itarget = itarget.to(DEVICE)
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
                fn += '_to_' + target_speaker
                fn = os.path.join(PATH_OUT, fn + '.wav')

                # Synthesize
                # print(str(nfiles+1)+'/'+str(len(fnlist))+'\t'+fn)
                sys.stdout.flush()
                synthesize(audio, fn, STRIDE, sr=SR, normalize=True)

                # Reset
                audio = []
                nfiles += 1
