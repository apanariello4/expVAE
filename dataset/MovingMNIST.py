import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
from einops import rearrange, repeat

from torchvision.transforms import RandomErasing

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.utils import get_project_root


def to_tensor(pic: np.ndarray) -> Tensor:
    assert isinstance(pic, np.ndarray)
    img = torch.from_numpy(pic).contiguous()
    return img.float().div(255)


class MovingMNIST(data.Dataset):
    def __init__(self, train: bool, anom: bool = False) -> None:
        super().__init__()
        self.train = train
        self.anom = anom
        root = get_project_root()
        train_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_train.npz')
        test_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_test.npz')
        anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom_test.npz')
        anom_label_path = Path(f'{root}/data/anom_moving_mnist/labels_anom_test.npy')

        assert train_path.exists() and test_path.exists()
        assert anom_path.exists() and anom_label_path.exists()

        if self.anom:
            self.data = np.load(anom_path)['anommnist']
            self.labels = np.load(anom_label_path)
        elif self.train:
            self.data = np.load(train_path)['anommnist']
        else:
            self.data = np.load(test_path)['anommnist']

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[index]

        sequence = to_tensor(sequence)
        sequence = sequence.transpose(0, 1)

        target = Tensor(self.labels[index]).type(torch.int) if self.anom else Tensor()

        return sequence, target

    def __len__(self) -> int:
        return len(self.data)


class MovingMNISTResNet(data.Dataset):
    """Dataset object of the MovingMNIST dataset for ResNets (classification).

        __get_item__ function returns a tuple of the form (sequence, target)
        where sequence is a Tensor of shape (sequence_length, channels, h, w),
        target is a Tensor of shape (1, ) and is 1 for sequences containing an anomaly,
        0 otherwise.
    """

    def __init__(self, train: bool, channels: int = 1) -> None:
        super().__init__()

        self.train = train
        self.channels = channels
        root = get_project_root()
        self.random_erasing = RandomErasing(p=1, scale=(0.05, 0.1), ratio=(0.8, 1.2), value=0.5)
        train_normal_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_train.npz')
        test_normal_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_test.npz')

        train_anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom_train.npz')
        # train_anom_labels_path = Path(f'{root}/data/anom_moving_mnist/labels_anom_train.npy')
        test_anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom_test.npz')
        # test_anom_label_path = Path(f'{root}/data/anom_moving_mnist/labels_anom_test.npy')

        if self.train:
            assert train_normal_path.exists() and train_anom_path.exists()
            data_normal = np.load(train_normal_path)['anommnist'][5000:, ]
            data_anom = np.load(train_anom_path)['anommnist']
        else:
            assert test_normal_path.exists() and test_anom_path.exists()
            data_normal = np.load(test_normal_path)['anommnist']
            data_anom = np.load(test_anom_path)['anommnist']

        self.data = np.concatenate((data_normal, data_anom), axis=0)
        self.targets = np.concatenate([
            np.zeros(len(data_normal), dtype=np.int64),
            np.ones(len(data_anom), dtype=np.int64)])

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[index]
        target = self.targets[index]
        # n_frames = 20
        # # random_seq = np.random.randint(0, sequence.shape[0] - n_frames + 1)
        # # sequence = sequence[random_seq:random_seq + n_frames]
        # random_frame = np.random.randint(0, n_frames - 2)
        sequence = to_tensor(sequence)
        # if target == 1:
        #     sequence[random_frame:random_frame + 2] = self.random_erasing(sequence[random_frame:random_frame + 2])
        if self.channels == 3:
            sequence = repeat(sequence, 't c h w -> t (3 c) h w')
        sequence = rearrange(sequence, 't c h w -> c t h w')

        return sequence, target

    def __len__(self) -> int:
        return len(self.data)


class MovingMNISTMIL(data.Dataset):
    def __init__(self, train: bool, channels: int = 1) -> None:
        super().__init__()

        self.train = train
        self.channels = channels
        root = get_project_root()
        self.random_erasing = RandomErasing(p=1, scale=(0.05, 0.1), ratio=(0.8, 1.2), value=0.5)
        train_normal_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_train.npz')
        test_normal_path = Path(f'{root}/data/anom_moving_mnist/anommnist_normal_test.npz')

        train_anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom_train.npz')
        train_anom_labels_path = Path(f'{root}/data/anom_moving_mnist/labels_anom_train.npy')
        test_anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom_test.npz')
        test_anom_labels_path = Path(f'{root}/data/anom_moving_mnist/labels_anom_test.npy')

        if self.train:
            assert train_normal_path.exists() and train_anom_path.exists()
            self.data_normal = np.load(train_normal_path)['anommnist'][5000:, ]
            self.data_anom = np.load(train_anom_path)['anommnist']
            self.anom_labels = np.load(train_anom_labels_path)
        else:
            assert test_normal_path.exists() and test_anom_path.exists()
            self.data_normal = np.load(test_normal_path)['anommnist']
            self.data_anom = np.load(test_anom_path)['anommnist']
            self.anom_labels = np.load(test_anom_labels_path)

        self.n_anom_frames = (self.anom_labels == 1).sum()
        self.n_frames = self.data_normal.shape[0] * self.data_normal.shape[1] + self.data_anom.shape[0] * self.data_anom.shape[1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        norm = self.data_normal[index]
        anom = self.data_anom[index]

        sequence = np.concatenate((norm, anom), axis=0)
        sequence = to_tensor(sequence)
        if self.channels == 3:
            sequence = repeat(sequence, 't c h w -> t (3 c) h w')
        sequence = rearrange(sequence, 't c h w -> c t h w')

        return torch.chunk(sequence, chunks=2, dim=1), self.anom_labels[index]

    def __len__(self) -> int:
        return len(self.data_normal)


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    ds = MovingMNISTMIL(train=False)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    a = next(iter(dl))
    print(a[0].shape, a[1].shape)
    print(a[1])
    # for i in range(a[0].shape[0]):
    #     torchvision.utils.save_image(a[0][i].transpose(0, 1), f'./a_{i}_{a[1][i]}.png', nrow=20)
