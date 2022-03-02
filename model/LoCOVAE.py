import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.linear import ResidualLinear
from model.block2d import Decoder2DBlock, Encoder2DBlock
from model.block3d import Encoder3DBlock, Decoder3DBlock
from model.block2p1d import Encoder2p1Block, Decoder2p1Block

from model.pixel_shuffle import PixelShuffle3d


class BCTHWtoBCHW(nn.Module):
    def __init__(self):
        super(BCTHWtoBCHW, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        # 16,1,20,64,64 -> 16*20,1,64,64
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        return x


class BCHWtoBCTHW(nn.Module):
    def __init__(self, batch_size: int):
        super(BCHWtoBCTHW, self).__init__()
        self.batch_size = batch_size

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[0] // self.batch_size
        x = x.reshape(self.batch_size, t, *x.shape[1:])
        x = x.permute(0, 2, 1, 3, 4)
        return x


class LoCOVAE(nn.Module):
    def __init__(self, latent_dim: int, batch_size: int, activation: str = 'relu'):
        super(LoCOVAE, self).__init__()

        act = {'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               'elu': nn.ELU(inplace=True),
               'silu': nn.SiLU(inplace=True)}[activation]

        self.latent_dim = latent_dim
        self.name = 'LoCOVAE'

        self.encoder = nn.Sequential(
            BCTHWtoBCHW(),
            Encoder2DBlock(in_channels=1, out_channels=16, stride=1, activation=act),
            Encoder2DBlock(in_channels=16, out_channels=32, stride=2, activation=act),
            Encoder2DBlock(in_channels=32, out_channels=64, stride=2, activation=act),  # 64,16,16

            BCHWtoBCTHW(batch_size),
            Encoder3DBlock(in_channels=64, out_channels=128, stride=2, activation=act),
            Encoder3DBlock(in_channels=128, out_channels=256, stride=2, activation=act),
            Encoder3DBlock(in_channels=256, out_channels=512, stride=2, activation=act),  # 512,3,2,2

            nn.Flatten(),
            ResidualLinear(512 * 3 * 2 * 2, latent_dim * 2),
        )

        # hidden => mu
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim)

        self.decoder = nn.Sequential(
            ResidualLinear(latent_dim, 512 * 3 * 2 * 2),

            nn.Unflatten(1, (512, 3, 2, 2)),

            Decoder3DBlock(in_channels=512, out_channels=256, use_pixel_shuffle=2, activation=act),
            Decoder3DBlock(in_channels=256, out_channels=128, use_pixel_shuffle=2, activation=act),
            Decoder3DBlock(in_channels=128, out_channels=64, upsample_shape=(20, 16, 16), activation=act),  # b,64,20,16,16
            BCTHWtoBCHW(),

            Decoder2DBlock(in_channels=64, out_channels=32, upscale_factor=2, activation=act),
            Decoder2DBlock(in_channels=32, out_channels=16, upscale_factor=2, activation=act),
            Decoder2DBlock(in_channels=16, out_channels=1, upscale_factor=1, activation=act),
            BCHWtoBCTHW(batch_size),

        )

        print(f'{self.name}\tModel parameters: {self.count_parameters:,}')

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def gen_from_noise(self, z: Tensor) -> Tensor:
        return self.decode(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = LoCOVAE(latent_dim=128, batch_size=16)
    x = torch.randn(16, 1, 20, 64, 64)

    y = model(x)

    print(y[0].shape)
