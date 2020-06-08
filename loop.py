import numpy as np
from tqdm.auto import tqdm


HALFLOGTWOPI = 0.5 * np.log(2 * np.pi).item()
LOSS_PARTS = ['nll', 'log_p', 'log_det']


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
