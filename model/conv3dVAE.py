import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


from model.vae_loss import VAELoss

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.linear import ResidualLinear
from model.block2d import Encoder2DBlock
from model.block2p1d import Decoder2p1Block, Encoder2p1Block
from model.block3d import Decoder3DBlock, Encoder3DBlock
from model.pixel_shuffle import PixelShuffle3d
from model.base_model import BaseModel


class Conv3dVAE(BaseModel):
    def __init__(self, latent_dim: int, activation: str = 'relu'):
        super(Conv3dVAE, self).__init__()

        act = {'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               'elu': nn.ELU(inplace=True),
               'silu': nn.SiLU(inplace=True)}[activation]

        self.latent_dim = latent_dim
        self.name = 'conv3dVAE'

        self.encoder = nn.Sequential(

            Encoder2p1Block(in_channels=1, out_channels=16, stride=1, activation=act),

            Encoder2p1Block(in_channels=16, out_channels=16, stride=2, activation=act),
            Encoder2p1Block(in_channels=16, out_channels=16, stride=1, activation=act),
            Encoder2p1Block(in_channels=16, out_channels=32, stride=2, activation=act),
            Encoder2p1Block(in_channels=32, out_channels=32, stride=1, activation=act),
            Encoder2p1Block(in_channels=32, out_channels=64, stride=2, activation=act),
            Encoder2p1Block(in_channels=64, out_channels=64, stride=1, activation=act),

            nn.AdaptiveAvgPool3d((1, 8, 8)),
            nn.Flatten(start_dim=1, end_dim=2),
            Encoder2DBlock(in_channels=64, out_channels=128, stride=2, activation=act),  # b,128,4,4

            nn.Flatten(),
            ResidualLinear(128 * 4 * 4, 1024, activation=act),
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, latent_dim)

        self.decoder = nn.Sequential(
            ResidualLinear(latent_dim, 128 * 4 * 4),

            nn.Unflatten(1, (64, 2, 4, 4)),

            Decoder2p1Block(in_channels=64, out_channels=64, upsample_t=2, activation=act),
            Decoder2p1Block(in_channels=64, out_channels=64, upsample_t=1, upsample_w_h=1, activation=act),
            Decoder2p1Block(in_channels=64, out_channels=64, upsample_t=2, activation=act),
            Decoder2p1Block(in_channels=64, out_channels=32, upsample_t=1, upsample_w_h=1, activation=act),
            Decoder2p1Block(in_channels=32, out_channels=32, upsample_shape=(20, 32, 32), activation=act),
            Decoder2p1Block(in_channels=32, out_channels=32, upsample_t=1, upsample_w_h=1, activation=act),
            Decoder2p1Block(in_channels=32, out_channels=16, upsample_shape=(20, 64, 64), activation=act),
            Decoder2p1Block(in_channels=16, out_channels=16, upsample_t=1, upsample_w_h=1, activation=act),

            nn.Conv3d(16, 1, kernel_size=1, stride=1, padding=0),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        self.image_size = x.size()[-1]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self,) -> Tensor:
        z = torch.randn(1, self.latent_dim).to(self._device)
        recon = self.decode(z)
        return recon.squeeze(0).transpose(0, 1)

    @staticmethod
    def get_loss_function(recon_function) -> nn.Module:
        return VAELoss(recon_function)


if __name__ == '__main__':

    model = Conv3dVAE(latent_dim=32)

    x = torch.randn(32, 1, 20, 64, 64)

    a, mu, logvar = model(x)

    print(a.shape)
