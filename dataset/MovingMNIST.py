from __future__ import print_function

from pathlib import Path
from typing import Tuple

import numpy as np
import torch.utils.data as data
from torch import Tensor


class MovingMNIST(data.Dataset):
    def __init__(self, train: bool, anom: bool = False) -> None:
        super().__init__()
        self.train = train
        self.anom = anom
        train_path = Path('/home/nello/expVAE/dataset/anommnist_train.npz')
        test_path = Path('/home/nello/expVAE/dataset/anommnist_test.npz')
        anom_path = Path('/home/nello/expVAE/data/anom_moving_mnist/anomnist_anom.npz')
        anom_label_path = Path('/home/nello/expVAE/data/anom_moving_mnist/labels_anomal.npy')

        if self.anom:
            self.data = np.load(anom_path)['anommnist']
            self.labels = np.load(anom_label_path)
        else:
            if self.train:
                self.data = np.load(train_path)['anommnist']
            else:
                self.data = np.load(test_path)['anommnist']

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        sequence = self.data[index]
        # sequence to tensor

        sequence = Tensor(sequence) / 255
        sequence = sequence.transpose(0, 1)

        if self.anom:
            target = self.labels[index]
        else:
            target = Tensor()

        return sequence, target

    def __len__(self) -> int:
        return len(self.data)


if __name__ == '__main__':
    import torchvision
    from torch.utils.data import DataLoader
    ds = MovingMNIST(
        train=True,)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    a = next(iter(dl))
    print("a")
    #torchvision.utils.save_image(a[0], './a.png')
