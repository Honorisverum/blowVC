import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import beta
from sklearn.utils import shuffle
from torch.utils.data import Dataset

EXTENSION = '.pt'


class DatasetVC(Dataset):
    def __init__(self, path_in, lchunk, stride, split='train', frame_energy_thres=0,
                 temp_jitter=False, seed=0, sampling_rate=16000, is_aug=False):
        assert split in ['train', 'valid']
        self.path_in = path_in
        self.lchunk = lchunk
        self.stride = stride
        self.temp_jitter = temp_jitter
        self.filenames = sorted(glob.glob(f'{path_in}/**/*.pt'))
        self.filenames = shuffle(self.filenames, random_state=seed)
        self.is_aug = is_aug
        self.augmentations = DataAugmentation()

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
                for ut in ut_del: del self.utterances[spk][ut]
            elif split == 'valid':
                ut_del = lutterances[:-isplit_ut]
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
            print('\rRead audio {:5.1f}%'.format(100 * (i + 1) / len(self.filenames)), end='')
            # Info
            spk, ut = self.filename_split(filename)
            ispk, iut = self.speakers[spk], self.utterances[spk][ut]
            # Load
            if spk not in duration:
                duration[spk] = 0
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
            self.indices[-1][2] = 1
        self.indices = torch.stack(self.indices)

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
        if self.is_aug:
            x = self.augmentations.emphasis(x, 0.2)
            x = self.augmentations.magnorm_flip(x, 1)
        return x, y


class DataAugmentation(object):
    def __init__(self, betaparam=0.2):
        self.betadist = beta.Beta(betaparam, betaparam)

    @staticmethod
    def _get_random_vector(size):
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
        alpha = val * (2 * self._get_random_vector(len(x)) - 1)
        sign = alpha.sign()
        alpha = alpha.abs()
        xorig = x.clone()
        mx = torch.ones_like(alpha)
        x[:, 1:] += sign * alpha * xorig[:, :-1]
        mx += alpha
        alpha *= alpha
        x[:, 2:] += sign * alpha * xorig[:, :-2]
        mx += alpha
        alpha *= alpha
        x[:, 3:] += sign * alpha * xorig[:, :-3]
        mx += alpha
        alpha *= alpha
        x[:, 4:] += sign * alpha * xorig[:, :-4]
        mx += alpha  # ; alpha*=alpha
        return x / mx

    def mixup(self, x):
        lamb = self.betadist.sample((len(x), 1)).to(x.device)
        lamb = torch.max(lamb, 1 - lamb)
        perm = torch.randperm(len(x)).to(x.device)
        return lamb * x + (1 - lamb) * x[perm]
