import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset

EXTENSION = '.pt'


class DataSet(Dataset):

    def __init__(self, path_in, lchunk, stride, split='train', trim=None,
                 split_utterances=True,
                 frame_energy_thres=0, temp_jitter=False,
                 select_speaker=None, select_file=None, seed=0, verbose=True,
                 store_in_ram=True, sampling_rate=16000):
        self.path_in = path_in
        self.lchunk = lchunk
        self.stride = stride
        self.temp_jitter = temp_jitter
        self.store_in_ram = store_in_ram
        trim = np.inf

        self.filenames = sorted(glob.glob(f'{path_in}/**/*.pt'))
        self.filenames = shuffle(self.filenames, random_state=seed)

        # Get speakers & utterances
        self.speakers, self.utterances = {}, defaultdict(dict)
        for fname in self.filenames:
            spk, ut = self.filename_split(fname)
            if spk not in self.speakers:
                self.speakers[spk] = len(self.speakers)
            if ut not in self.utterances[spk]:
                self.utterances[spk][ut] = len(self.utterances[spk])
        self.maxspeakers = len(self.speakers)

        # Split
        for spk in self.speakers:
            lutterances = list(self.utterances[spk].keys())
            lutterances.sort()
            lutterances = shuffle(lutterances, random_state=seed)
            isplit_ut = int(len(lutterances) * 0.1)

            if split == 'train':
                ut_del = lutterances[-isplit_ut:]
            elif split == 'valid':
                ut_del = lutterances[:-isplit_ut]
            if split_utterances:
                for ut in ut_del: del self.utterances[spk][ut]

        # Filter filenames by speaker and utterance
        filenames_new = []
        for filename in self.filenames:
            spk, ut = self.filename_split(filename)
            if spk in self.speakers and ut in self.utterances[spk]:
                filenames_new.append(filename)
        self.filenames = filenames_new

        # Indices!
        self.audios = [None] * len(self.filenames)
        self.indices = []
        duration = {}
        for i, filename in enumerate(self.filenames):
            if verbose:
                print('\rRead audio {:5.1f}%'.format(100 * (i + 1) / len(self.filenames)), end='')
            # Info
            spk, ut = self.filename_split(filename)
            ispk, iut = self.speakers[spk], self.utterances[spk][ut]
            # Load
            if spk not in duration:
                duration[spk] = 0
            if duration[spk] >= trim:
                continue
            x = torch.load(filename)
            self.audios[i] = x.clone()
            x = x.float()
            # Process
            for j in range(0, len(x), stride):
                if j + self.lchunk >= len(x):
                    xx = x[j:]
                else:
                    xx = x[j:j + self.lchunk]
                if (xx.pow(2).sum() / self.lchunk).sqrt().item() >= frame_energy_thres:
                    info = [i, j, 0, ispk, iut]
                    self.indices.append(torch.LongTensor(info))
                duration[spk] += stride / sampling_rate
                if duration[spk] >= trim:
                    break
            self.indices[-1][2] = 1
        self.indices = torch.stack(self.indices)

        # Print
        print(f' Loaded {split}: {len(self.speakers)} spk, '
              f'{len(self.filenames)} ut, {len(self.indices)} frames')

    @staticmethod
    def filename_split(path):
        return Path(path).stem.split('_')[0:2]

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self, idx):
        i, j, last, ispk, ichap = self.indices[idx, :]
        # Load file
        tmp = self.audios[i]
        # Temporal jitter
        if self.temp_jitter:
            j = j + np.random.randint(-self.stride // 2, self.stride // 2)
            if j < 0:
                j = 0
            elif j > len(tmp) - self.lchunk:
                j = np.max([0, len(tmp) - self.lchunk])
        # Get frame
        if j + self.lchunk > len(tmp):
            x = tmp[j:].float()
            x = torch.cat([x, torch.zeros(self.lchunk - len(x))])
        else:
            x = tmp[j:j + self.lchunk].float()
        # Get info
        y = torch.LongTensor([i, j, last, ispk, ichap])
        return x, y


class DataAugmentation(object):

    def __init__(self, device, betaparam=0.2):
        self.device = device
        self.betadist = torch.distributions.beta.Beta(betaparam, betaparam)
        return

    def _get_random_vector(self, size):
        if self.device != torch.device('cpu'):
            return torch.cuda.FloatTensor(size, 1).uniform_()
        return torch.rand(size, 1)

    def magnorm(self, x, val):
        return x * val * (self._get_random_vector(len(x)) / (x.abs().max(1, keepdim=True)[0] + 1e-7))

    def flip(self, x):
        return x * (self._get_random_vector(len(x)) - 0.5).sign()

    def magnorm_flip(self, x, val):
        return x * val * ((2 * self._get_random_vector(len(x)) - 1) / (x.abs().max(1, keepdim=True)[0] + 1e-7))

    def compress(self, x, val):
        return x.sign() * (x.abs() ** (1 + val * (2 * self._get_random_vector(len(x)) - 1)))

    def noiseu(self, x, val):
        return x + val * self._get_random_vector(len(x)) * (2 * torch.rand_like(x) - 1)

    def noiseg(self, x, val):
        return x + val * self._get_random_vector(len(x)) * torch.randn_like(x)

    def emphasis(self, x, val):
        # http://www.fon.hum.uva.nl/praat/manual/Sound__Filter__de-emphasis____.html
        # (unrolled and truncated version; performs normalization but might be better to re-normalize afterwards)
        alpha = val * (2 * self._get_random_vector(len(x)) - 1)
        sign = alpha.sign()
        alpha = alpha.abs()
        xorig = x.clone();
        mx = torch.ones_like(alpha)
        x[:, 1:] += sign * alpha * xorig[:, :-1];
        mx += alpha;
        alpha *= alpha
        x[:, 2:] += sign * alpha * xorig[:, :-2];
        mx += alpha;
        alpha *= alpha
        x[:, 3:] += sign * alpha * xorig[:, :-3];
        mx += alpha;
        alpha *= alpha
        x[:, 4:] += sign * alpha * xorig[:, :-4];
        mx += alpha  # ; alpha*=alpha
        return x / mx

    def mixup(self, x):
        lamb = self.betadist.sample((len(x), 1)).to(x.device)
        lamb = torch.max(lamb, 1 - lamb)
        perm = torch.randperm(len(x)).to(x.device)
        return lamb * x + (1 - lamb) * x[perm]
