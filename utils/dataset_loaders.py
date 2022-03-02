import argparse
from typing import Tuple
from pathlib import Path

import torchvision.transforms as transforms
from dataset.OneClassFMNIST import OneFMNIST
from dataset.OneClassMNIST import OneMNIST
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


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


def load_moving_mnist(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:

    from dataset.MovingMNIST import MovingMNIST

    # data_path = Path(args.dataset_path) / 'MovingMNIST'

    train = MovingMNIST(train=True)
    test = MovingMNIST(train=False)
    anom = MovingMNIST(train=False, anom=True)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(
        test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    anom_loader = DataLoader(
        anom, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, test_loader, anom_loader
