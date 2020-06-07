import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import DataSet, DataAugmentation
from model import Model

warnings.filterwarnings('ignore')
HALFLOGTWOPI = 0.5 * np.log(2 * np.pi).item()


def gaussian_log_p(x, mu=None, log_sigma=None):
    if mu is None or log_sigma is None:
        return -HALFLOGTWOPI - 0.5 * (x ** 2)
    return -HALFLOGTWOPI - log_sigma - 0.5 * ((x - mu) ** 2) / torch.exp(2 * log_sigma)


def loss_flow_nll(z, log_det):
    # size of: z = sbatch * lchunk
    #          log_det = sbatch
    _, size = z.size()

    log_p = gaussian_log_p(z).sum(1)

    nll = -log_p - log_det

    log_det /= size
    log_p /= size
    nll /= size

    log_det = log_det.mean()
    log_p = log_p.mean()
    nll = nll.mean()

    return nll, np.array([nll.item(), log_p.item(), log_det.item()], dtype=np.float32)


def init_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)


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
# LR = 5e-3
LR_TRESH = 1e-4
LR_PATIENCE = 7
LR_FACTOR = 0.2
LR_RESTARTS = 3
BASE_FN_OUT = 'blow'
init_seed(SEED)

print('Loading data...')
dataset_train = DataSet(PATH_DATA, LCHUNK, STRIDE, split='train', sampling_rate=SR,
                        trim=TRIM, frame_energy_thres=FRAME_ENERGY_THRES,
                        temp_jitter=True, seed=SEED)
dataset_valid = DataSet(PATH_DATA, LCHUNK, STRIDE, split='valid', sampling_rate=SR,
                        trim=TRIM, frame_energy_thres=FRAME_ENERGY_THRES,
                        temp_jitter=False, seed=SEED)

loader_train = DataLoader(dataset_train, batch_size=SBATCH, shuffle=True, drop_last=True, num_workers=NWORKERS)
loader_valid = DataLoader(dataset_valid, batch_size=SBATCH, shuffle=False, num_workers=NWORKERS)
batch_data_augmentation = DataAugmentation(DEVICE)


def batch_loop(epoch, eval, loader):
    if eval:
        print(f'Eval epoch: {epoch}')
        model.eval()
    else:
        print(f'Train epoch: {epoch}')
        model.train()
    cum_losses, cum_num = np.zeros(3), 0
    LOSS_PARTS = ['nll', 'log_p', 'log_det']
    pbar = tqdm(loader, total=len(loader), leave=False)
    for x, info in pbar:
        # Prepare data
        s = info[:, 3].to(DEVICE)
        x = x.to(DEVICE)
        if not eval:
            x = batch_data_augmentation.emphasis(x, 0.2)
            x = batch_data_augmentation.magnorm_flip(x, 1)

        # Forward
        z, log_det = model.forward(x, s)
        loss, losses = loss_flow_nll(z, log_det)

        # Backward
        if not eval:
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Report/print
        cum_losses += losses * len(x)
        cum_num += len(x)
        pbar.set_description(
            " | ".join([f"{prt}:{val:.2f}" for prt, val in zip(LOSS_PARTS, cum_losses / cum_num)])
        )

    cum_losses /= cum_num
    print(" | ".join([f"{prt}:{val:.2f}" for prt, val in zip(LOSS_PARTS, cum_losses)]))
    return cum_losses[0], cum_losses


def load_best_model(model):
    model = load_stuff(BASE_FN_OUT, model)
    model = model.to(DEVICE)
    if NGPUS > 1: return torch.nn.DataParallel(model)
    return model


def get_optimizer(lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


print('Init model and tools...')
# model = Model(sqfactor=2, nblocks=2, nflows=3, ncha=256, ntargets=dataset_train.maxspeakers)
model = Model(sqfactor=2, nblocks=8, nflows=12, ncha=512, ntargets=dataset_train.maxspeakers)
model.to(DEVICE)
optim = get_optimizer(LR)
losses = {'train': [], 'valid': []}
loss_best = np.inf
save_stuff(BASE_FN_OUT, model=model)

if NGPUS > 1:
    print(f'[Using {NGPUS} GPUs]...')
    model = torch.nn.DataParallel(model)

print('Train...')
lr, patience, restarts = LR, LR_PATIENCE, LR_RESTARTS

for epoch in range(N_EPOCHS):
    # train loop
    _, losses_train = batch_loop(epoch, False, loader_train)
    losses['train'].append(losses_train)
    # valid loop
    with torch.no_grad():
        loss, losses_valid = batch_loop(epoch, True, loader_valid)
        losses['valid'].append(losses_valid)
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
            print(f'lr={lr:.4f}', end='')
            optim = get_optimizer(lr)
            patience = LR_PATIENCE
