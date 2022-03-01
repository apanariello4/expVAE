import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.utils.data as data
from torch import Tensor
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from utils.utils import get_project_root


class MovingMNIST(data.Dataset):
    def __init__(self, train: bool, anom: bool = False) -> None:
        super().__init__()
        self.train = train
        self.anom = anom
        root = get_project_root()
        train_path = Path(f'{root}/data/anom_moving_mnist/anommnist_train.npz')
        test_path = Path(f'{root}/data/anom_moving_mnist/anommnist_test.npz')
        anom_path = Path(f'{root}/data/anom_moving_mnist/anomnist_anom.npz')
        anom_label_path = Path(f'{root}/data/anom_moving_mnist/labels_anomal.npy')

        if self.anom:
            self.data = np.load(anom_path)['anommnist']
            self.labels = np.load(anom_label_path)
        elif self.train:
            self.data = np.load(train_path)['anommnist']
        else:
            self.data = np.load(test_path)['anommnist']

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[index]

        sequence = Tensor(sequence) / 255
        sequence = sequence.transpose(0, 1)

        target = Tensor(self.labels[index]).type(torch.int) if self.anom else Tensor()

        return sequence, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    ds = MovingMNIST(
        train=True)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    a = next(iter(dl))
    print(a[0].shape, a[1].shape)
    #torchvision.utils.save_image(a[0], './a.png')
