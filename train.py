import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import argparse
from tqdm.auto import tqdm

from data import DatasetVC
from models.blow import Model

warnings.filterwarnings('ignore')
HALFLOGTWOPI = 0.5 * np.log(2 * np.pi).item()
LOSS_PARTS = ['nll', 'log_p', 'log_det']


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


def load_model(name, device='cpu'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = torch.load(f"weights/{name}.pt", map_location=device)
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Audio synthesis script')
    parser.add_argument('--model_fname', default='blow', type=str, required=True)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    return args


def loss_flow(z, log_det):
    def gaussian_log_p(x):
        return -HALFLOGTWOPI - 0.5 * (x ** 2)

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


def loop(model, mode, loader, optim, device):
    if mode == 'eval': model.eval()
    elif mode == 'train': model.train()
    cum_losses, cum_num = np.zeros(3), 0

    pbar = tqdm(loader, total=len(loader), leave=False)
    for x, info in pbar:
        # device
        s = info[:, 3].to(device)
        x = x.to(device)
        # Forward
        loss, losses = loss_flow(*model(x, s))
        # Backward
        if mode == 'train':
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
    nll, _, _ = cum_losses
    return nll, model, optim


def build_loaders():
    print('Loading data...')
    dataset_train = DatasetVC(PATH_DATA, LCHUNK, STRIDE, split='train', sampling_rate=SR,
                              frame_energy_thres=FRAME_ENERGY_THRES, temp_jitter=True, seed=args.seed, is_aug=True)
    dataset_valid = DatasetVC(PATH_DATA, LCHUNK, STRIDE, split='valid', sampling_rate=SR,
                              frame_energy_thres=FRAME_ENERGY_THRES, temp_jitter=False, seed=args.seed, is_aug=False)
    loader_train = DataLoader(dataset_train, batch_size=SBATCH, shuffle=True, drop_last=True, num_workers=NWORKERS)
    loader_valid = DataLoader(dataset_valid, batch_size=SBATCH, shuffle=False, num_workers=NWORKERS)


def load_best_model(model):
    model = load_stuff(BASE_FN_OUT, model)
    model = model.to(DEVICE)
    if NGPUS > 1: return torch.nn.DataParallel(model)
    return model


def build_tools(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)


def run_train(args):

    print('Setup args...')
    PATH_DATA = 'data'
    TRIM = None
    LCHUNK = 4096
    STRIDE = LCHUNK
    SR = 16000
    FRAME_ENERGY_THRES = 0.025
    NGPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    SBATCH = 38 * NGPUS
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

    print('Loading data...')
    loader_train, loader_valid = build_loaders()

    print('Init model and tools...')
    model = Model(sqfactor=2, nblocks=8, nflows=12, ncha=512, ntargets=loader_train.dataset.maxspeakers)
    model.to(DEVICE)
    optim, scheduler = build_tools(model, LR)
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
                optim = build_tools(lr)
                patience = LR_PATIENCE
        print()


if __name__ == '__main__':
    args = get_args()
    init_seed(args.seed)
    run_train(args)
