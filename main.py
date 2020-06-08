import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn

from loop import loop
from data import DatasetVC
from model import Model

warnings.filterwarnings('ignore')
HALFLOGTWOPI = 0.5 * np.log(2 * np.pi).item()


def init_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)


def save_stuff(basename, model=None, epoch=None):
    basename = 'weights/' + basename
    suf = '' if epoch is None else str(epoch)
    if model is not None:
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel)
                   else model.state_dict(), basename + suf + '.model.pt')


def load_stuff(basename, model, device='cpu'):
    basename = 'weights/' + basename
    state = torch.load(basename + '.model.pt', map_location=device)
    model.load_state_dict(state, strict=True)
    return model


print('Setup args...')
PATH_DATA = 'data'
TRIM = None
LCHUNK = 4096
STRIDE = LCHUNK
SR = 16000
FRAME_ENERGY_THRES = 0.025
NGPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
SBATCH = 38 * NGPUS
SEED = 0
NWORKERS = 0
AUGMENT = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 999
LR = 1e-4
LR_TRESH = 1e-4
LR_PATIENCE = 7
LR_FACTOR = 0.2
LR_RESTARTS = 3
BASE_FN_OUT = 'blow'
init_seed(SEED)

print('Loading data...')
dataset_train = DatasetVC(PATH_DATA, LCHUNK, STRIDE, split='train', sampling_rate=SR,
                          frame_energy_thres=FRAME_ENERGY_THRES, temp_jitter=True, seed=SEED, is_aug=True)
dataset_valid = DatasetVC(PATH_DATA, LCHUNK, STRIDE, split='valid', sampling_rate=SR,
                          frame_energy_thres=FRAME_ENERGY_THRES, temp_jitter=False, seed=SEED, is_aug=False)
loader_train = DataLoader(dataset_train, batch_size=SBATCH, shuffle=True, drop_last=True, num_workers=NWORKERS)
loader_valid = DataLoader(dataset_valid, batch_size=SBATCH, shuffle=False, num_workers=NWORKERS)


def load_best_model(model):
    model = load_stuff(BASE_FN_OUT, model)
    model = model.to(DEVICE)
    if NGPUS > 1: return torch.nn.DataParallel(model)
    return model


def get_optimizer(lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


print('Init model and tools...')
# model = Model(sqfactor=2, nblocks=4, nflows=6, ncha=512, ntargets=dataset_train.maxspeakers)
model = Model(sqfactor=2, nblocks=8, nflows=12, ncha=512, ntargets=dataset_train.maxspeakers)
model.to(DEVICE)
optim = get_optimizer(LR)
loss_best = np.inf
save_stuff(BASE_FN_OUT, model=model)

if NGPUS > 1:
    print(f'[Using {NGPUS} GPUs]...')
    model = torch.nn.DataParallel(model)

print('Train...')
lr, patience, restarts = LR, LR_PATIENCE, LR_RESTARTS

for epoch in range(N_EPOCHS):
    # train/valid loop
    _, model, optim = loop(model, 'train', loader_train, optim, DEVICE)
    with torch.no_grad():
        loss, model, optim = loop(model, 'valid', loader_valid, optim, DEVICE)
    # Control stall
    if np.isnan(loss) or loss > 1000:
        patience = 0
        loss = np.inf
        model = load_best_model(model)
    # Best model?
    if loss < loss_best * (1 + LR_TRESH):
        loss_best = loss
        patience = LR_PATIENCE
        save_stuff(BASE_FN_OUT, model=model, epoch=epoch)
    else:
        # Learning rate annealing or exit
        patience -= 1
        if patience <= 0:
            restarts -= 1
            if restarts < 0:
                print('End...')
                break
            lr *= LR_FACTOR
            print(f'lr={lr:.7f}')
            optim = get_optimizer(lr)
            patience = LR_PATIENCE
    print()
