import argparse
from logging import Logger
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
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
    parser.add_argument('--alpha-scheduler', type=str, default='step', choices=['warmup', 'cyclic'])
    # LR
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-steps', type=int, default=3)
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine'])

    parser.add_argument('--activation', type=str, default='relu',
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


def apply_jet_color_map(img: np.ndarray) -> np.ndarray:
    """
    Apply jet color map to an image
    """
    img = img.mean(axis=0)
    img = img - np.min(img)
    img = img / np.max(img)
    img = cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET)
    return img


def save_cam(image: np.ndarray, filename: str,
             gcam: np.ndarray, gcam_loss=None) -> None:
    image = np.stack(([image] * 3), axis=-1)

    # latent space gcam
    gcam = apply_jet_color_map(gcam)
    gcam = np.asarray(gcam, dtype=np.float) + np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    imgs = np.concatenate((image, gcam), axis=0)

    # loss gcam
    if gcam_loss:
        gcam_loss = apply_jet_color_map(gcam_loss)
        gcam_loss = np.asarray(gcam_loss, dtype=np.float) + \
            np.asarray(image, dtype=np.float)
        gcam_loss = 255 * gcam_loss / np.max(gcam_loss)
        gcam_loss = np.uint8(gcam_loss)
        imgs = np.concatenate((imgs, gcam_loss), axis=1)

    cv2.imwrite(filename, imgs[0])


def get_project_root() -> Path:
    return Path(__file__).parent.parent
