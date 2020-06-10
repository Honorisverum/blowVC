import argparse, os
import numpy as np
import torch
import torch.utils.data
from copy import deepcopy
from tqdm.auto import tqdm
from scipy.io import wavfile
from torch.backends import cudnn
from torch.utils.data import DataLoader
import warnings

from data import DatasetVC


def init_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)


def load_model(name, device='cpu'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = torch.load(f"weights/{name}.pt", map_location=device)
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


def get_args():
    parser = argparse.ArgumentParser(description='Audio synthesis script')
    parser.add_argument('--model_fname', default='blow', type=str, required=True)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    return args


def run_synthesize(args):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LCHUNK = 4096
    STRIDE = LCHUNK // 2
    FORCE_TARGET_SPEAKER = 'BillGates'
    SBATCH = 256
    PATH_DATA = 'data'
    PATH_OUT = 'synth'

    print('Load model...')
    model = load_model(args.model_fname)
    model = model.to(DEVICE)
    window = torch.hann_window(4096).view(1, -1)

    # Input data
    print('Load data...')
    dataset = DatasetVC(PATH_DATA, LCHUNK, STRIDE, split='valid', seed=args.seed)
    loader = DataLoader(dataset, batch_size=SBATCH, shuffle=False, num_workers=0)
    speakers = deepcopy(dataset.speakers)

    print("Preparing out filenames...")
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

    print('Synthesize...')
    model.precalc_matrices('on')
    model.eval()
    audio, nfiles = [], 0
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
            # make audio
            for n in range(len(x)):
                audio.append(x[n])
                i, j, last, ispk, iut = info[n]
                if last == 1:
                    fn, source_speaker, target_speaker = fnlist[nfiles]
                    _, fn = os.path.split(fn)
                    # Synthesize
                    synthesize(audio, filename=f"{PATH_OUT}/{fn}_to_{target_speaker}.wav", stride=STRIDE)
                    # Reset
                    audio = []
                    nfiles += 1


if __name__ == '__main__':
    args = get_args()
    init_seed(args.seed)
    run_synthesize(args)
