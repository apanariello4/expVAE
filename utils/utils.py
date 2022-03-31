import argparse
import os
import random
from pathlib import Path

import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import Tensor


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--dataset-path', type=str, default='./data/mnist')
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=512)
    # KLD ALPHA
    parser.add_argument('--alpha-warmup', type=int, default=10)  # if 0 no warmup
    parser.add_argument('--alpha-min', type=float, default=0.0)
    parser.add_argument('--alpha-max', type=float, default=1.0)
    parser.add_argument('--alpha-scheduler', type=str, default='warmup', choices=['warmup', 'cyclic'])
    # LR
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-steps', type=int, default=3)
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])

    parser.add_argument('--activation', type=str, default='elu',
                        choices=['relu', 'elu', 'silu', 'leakyrelu'])
    parser.add_argument('--recon-func', type=str, default='bce', choices=['mse', 'bce'])
    parser.add_argument('--model', type=str, default='conv3d', choices=['loco', 'conv3d', 'vrnn', 'bivrnn', 'dsvae'])

    parser.add_argument('--resume', type=str)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--one-class', type=int, default=3)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist'])
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--name', type=str)
    parser.add_argument('--attention', action='store_true')

    return parser.parse_args()


def save_checkpoint(model, epoch: int, optimizer: torch.optim,
                    is_last: bool, args: argparse.Namespace, time) -> None:
    outdir = args.ckpt_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    experiment = args.name + '_' if args.name else ''
    base_path = os.path.join(
        outdir, f'{experiment}_{model.name}_{time}')
    checkpoint_file = base_path + '_chkp.pth'
    best_file = base_path + '_best.pth'
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if is_last:
        print(f"Saving checkpoint in {checkpoint_file}")
        torch.save(state, checkpoint_file)
    else:
        print(f"Saving new best model in {best_file}")
        torch.save(state, best_file)


def get_alpha_scheduler(args: argparse.Namespace) -> torch.Tensor:
    alpha = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_warmup)
    if args.alpha_scheduler == 'warmup':
        alpha = torch.cat((alpha, torch.full((args.epochs - args.alpha_warmup,), fill_value=args.alpha_max)))
    elif args.alpha_scheduler == 'cyclic':
        cycles = args.epochs // (args.alpha_warmup * 2)
        if cycles < 4:
            print(f'Cycles should be higher, cycles={cycles}')
        alpha = torch.cat((alpha, torch.full((args.alpha_warmup,), fill_value=args.alpha_max)))
        alpha = alpha.repeat(args.epochs // (args.alpha_warmup * 2))
        if alpha.shape[0] < args.epochs:
            alpha = torch.cat((alpha, torch.full((args.epochs - alpha.shape[0],), fill_value=args.alpha_max)))

    return alpha


def deterministic_behavior(seed: int = 1) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def apply_jet_color_map_to_seq(img: np.ndarray, min, max) -> np.ndarray:
    """
    Apply jet color map to a sequence.
    The map is normalized sequence level to [0, 1].

    img: numpy array of shape (T, H, W)
    """
    imgs = np.zeros(img.shape + (3,))
    img -= img.min()
    img /= img.max()
    img = np.expand_dims(img, axis=-1)
    for i in range(img.shape[0]):
        if img[i].max() > 0:
            imgs[i] = cv2.applyColorMap(np.uint8(255 * img[i]), cv2.COLORMAP_JET)
    return imgs


def save_cam(image: np.ndarray, filename: str,
             gcam: np.ndarray, reconstruction: np.ndarray, gcam_loss=None) -> None:
    image = np.stack(([image] * 3), axis=-1)
    #reconstruction = np.uint8(255 * reconstruction / np.max(reconstruction))
    reconstruction = np.stack(([reconstruction] * 3), axis=-1)
    batch_min = np.min(gcam)
    batch_max = np.max(gcam)
    if gcam.ndim == 4:
        jet_color = np.zeros(gcam.shape + (3,))
        for i in range(gcam.shape[0]):
            jet_color[i] = apply_jet_color_map_to_seq(gcam[i], batch_min, batch_max)
        gcam = jet_color
    else:
        gcam = apply_jet_color_map_to_seq(gcam)

    # Apply gcam on top of original image

    gcam = np.asarray(gcam, dtype=np.float) + np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)

    # Apply gcam on top of reconstruction
    gcam_recon = np.asarray(gcam, dtype=np.float) + np.asarray(reconstruction, dtype=np.float)
    reconstruction = np.uint8(255 * reconstruction / np.max(reconstruction))
    gcam_recon = np.uint8(255 * gcam_recon / np.max(gcam_recon))

    if gcam.ndim == 5:
        gcam = einops.rearrange(gcam, 't seq h w c-> (t seq) h w c')
        gcam_recon = einops.rearrange(gcam_recon, 't seq h w c-> (t seq) h w c')

    imgs = np.concatenate((image, gcam), axis=0)

    # loss gcam
    if gcam_loss:
        gcam_loss = apply_jet_color_map_to_seq(gcam_loss)
        gcam_loss = np.asarray(gcam_loss, dtype=np.float) + \
            np.asarray(image, dtype=np.float)
        gcam_loss = 255 * gcam_loss / np.max(gcam_loss)
        gcam_loss = np.uint8(gcam_loss)
        imgs = np.concatenate((imgs, gcam_loss), axis=1)

    # imgs = np.concatenate((imgs, reconstruction), axis=0)

    # imgs = np.concatenate((imgs, gcam_recon), axis=0)
    #imgs in [0, 255]
    grid = torchvision.utils.make_grid(torch.from_numpy(imgs).transpose(-1, 1), nrow=image.shape[0])
    cv2.imwrite(filename, grid.transpose(0, -1).numpy())


def get_project_root() -> Path:
    return Path(__file__).parent.parent
