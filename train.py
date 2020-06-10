import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import argparse
import wandb
from torch.optim.lr_scheduler import ExponentialLR
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


def save_model(model_fname, model):
    torch.save(model.module if isinstance(model, torch.nn.DataParallel)
               else model, f"weights/{model_fname}.pt")


def load_model(model_fname, device, multigpu):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = torch.load(f"weights/{model_fname}.pt", map_location=device)
    model = model.to(device)
    if multigpu:
        return torch.nn.DataParallel(model)
    return model


def get_args():
    parser = argparse.ArgumentParser(description='Audio synthesis script')
    parser.add_argument('--model_fname', default='blow', type=str, required=True)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    return args


def setup_device(model, ngpus):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ngpus > 1:
        print(f'[Using {ngpus} GPUs]...')
        return torch.nn.DataParallel(model), device
    return model, device


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


def loop(model, mode, loader, optim, scheduler, device):
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
        wandb.log({f"{mode}_{prt}": val for prt, val in zip(LOSS_PARTS, cum_losses / cum_num)})

    if mode == 'train': scheduler.step()
    cum_losses /= cum_num
    print(" | ".join([f"{prt}:{val:.2f}" for prt, val in zip(LOSS_PARTS, cum_losses)]), end='\n\n'*(mode == 'eval'))
    wandb.log({f"overall_{mode}_{prt}": val for prt, val in zip(LOSS_PARTS, cum_losses)})
    nll, _, _ = cum_losses
    return nll, model, optim, scheduler


def build_loaders(path_data, lchunk, stride, sr, frame_energy, n_workers, sbatch, seed):
    dataset_train = DatasetVC(path_data, lchunk, stride, split='train', sampling_rate=sr,
                              frame_energy_thres=frame_energy, temp_jitter=True, seed=seed, is_aug=True)
    dataset_valid = DatasetVC(path_data, lchunk, stride, split='valid', sampling_rate=sr,
                              frame_energy_thres=frame_energy, temp_jitter=False, seed=seed, is_aug=False)
    loader_train = DataLoader(dataset_train, batch_size=sbatch, shuffle=True, drop_last=True, num_workers=n_workers)
    loader_valid = DataLoader(dataset_valid, batch_size=sbatch, shuffle=False, num_workers=n_workers)
    return loader_train, loader_valid, dataset_train.maxspeakers


def build_tools(model, lr, gamma):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optim, gamma)
    return optim, scheduler


def run_train(args):

    PATH_DATA = 'data'
    LCHUNK = 4096
    STRIDE = LCHUNK
    SR = 16000
    FRAME_ENERGY = 0.025
    NGPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
    SBATCH = 38 * NGPUS
    NWORKERS = 0
    GAMMA = 0.98
    N_EPOCHS = 500
    LR = 1e-4

    print('Loading data...')
    loader_train, loader_valid, maxspeakers = \
        build_loaders(PATH_DATA, LCHUNK, STRIDE, SR, FRAME_ENERGY, NWORKERS, SBATCH, args.seed)

    print('Init model and tools...')
    model = Model(sqfactor=2, nblocks=8, nflows=12, ncha=512, ntargets=maxspeakers)
    optim, scheduler = build_tools(model, LR, GAMMA)
    loss_best = np.inf

    print('Setup device...')
    model, device = setup_device(model, NGPUS)

    print('Train...')
    wandb.init(config=args)
    for epoch in range(N_EPOCHS):
        print(f"Starting {epoch} epoch...")
        _, model, optim, scheduler = loop(model, 'train', loader_train, optim, scheduler, device)
        with torch.no_grad():
            loss, model, optim, scheduler = loop(model, 'valid', loader_valid, optim, scheduler, device)
        if loss < loss_best:
            loss_best = loss
            save_model(args.model_fname, model)


if __name__ == '__main__':
    args = get_args()
    init_seed(args.seed)
    run_train(args)
