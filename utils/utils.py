import argparse
from email.policy import default
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset.OneClassMNIST import OneMNIST
from dataset.OneClassFMNIST import OneFMNIST
from dataset.MovingMNIST import MovingMNIST
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset-path', type=str, default='./data/mnist')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--alpha-warmup', type=int, default=10)
    parser.add_argument('--resume', type=str, choices=['best', 'last'])
    parser.add_argument('--one-class', type=int, default=3)
    parser.add_argument('--dataset', default='fmnist', choices=['mnist', 'fmnist'])
    return parser.parse_args()


def save_checkpoint(model, epoch: int, optimizer: torch.optim,
                    distribution: Tuple[Tensor, Tensor],
                    is_last: bool, outdir: str, args: argparse.Namespace) -> None:
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    dataset = args.dataset
    if model.name == 'conv3dVAE':
        dataset = 'moving'
    base_path = os.path.join(
        outdir, f'{dataset}_{model.name}_{args.latent_dim}_')
    checkpoint_file = base_path + 'checkpoint.pth'
    best_file = base_path + 'model_best.pth'
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mu_avg': distribution[0],
        'var_avg': distribution[1],
        'dataset': args.dataset,
    }
    if is_last:
        print("Saving checkpoint")
        torch.save(state, checkpoint_file)
    else:
        print("Saving new best model")
        torch.save(state, best_file)


def load_mnist(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.1307], [0.3081])
    ])

    train_dataset = MNIST(
        args.dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(
        args.dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    return train_loader, test_loader


def load_mnist_one_class(class_id: int,
                         args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.1307], [0.3081])
    ])

    data_path = Path(args.dataset_path) / 'OneMNIST'

    train = OneMNIST(
        data_path, one_class=class_id, train=True, download=True, transform=mnist_transform)
    test = OneMNIST(
        data_path, one_class=class_id, train=False, download=True, transform=mnist_transform)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader


def load_fmnist_one_class(class_id: int,
                          args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:

    fmnist_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_path = Path(args.dataset_path) / 'FMNIST'

    train = OneFMNIST(
        data_path, one_class=class_id, train=True, download=True, transform=fmnist_transform)
    test = OneFMNIST(
        data_path, one_class=class_id, train=False, download=True, transform=fmnist_transform)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader


def load_moving_mnist(
        args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:

    data_path = Path(args.dataset_path) / 'MovingMNIST'

    train = MovingMNIST(train=True)
    test = MovingMNIST(train=False)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader


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
             gcam: np.ndarray, gcam_loss) -> None:
    image = np.stack(([image] * 3), axis=2)

    # latent space gcam
    gcam = apply_jet_color_map(gcam)
    gcam = np.asarray(gcam, dtype=np.float) + np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    imgs = np.concatenate((image, gcam), axis=1)

    # loss gcam
    gcam_loss = apply_jet_color_map(gcam_loss)
    gcam_loss = np.asarray(gcam_loss, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam_loss = 255 * gcam_loss / np.max(gcam_loss)
    gcam_loss = np.uint8(gcam_loss)
    imgs = np.concatenate((imgs, gcam_loss), axis=1)

    cv2.imwrite(filename, imgs)
