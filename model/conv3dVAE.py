import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.block2d import EncoderBasic2DBlock
from model.block3d import EncoderBasic3DBlock, DecoderBasic3DBlock

from model.pixel_shuffle import PixelShuffle3d


class DebugConv3dVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super(DebugConv3dVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),
        )

        # flatten: (1,128,7,7) -> (1,128*7*7) = (1,6272)
        # hidden => mu
        self.fc1 = nn.Linear(4096, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(4096, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 64, 64)),
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x[:, :, 0, :, :])
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x).unsqueeze(1).repeat((1, 1, 20, 1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def gen_from_noise(self, z):
        return self.decode(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Conv3dVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super(Conv3dVAE, self).__init__()

        self.latent_dim = latent_dim
        self.name = 'conv3dVAE'

        self.encoder1 = nn.Sequential(
            EncoderBasic3DBlock(in_channels=1, out_channels=16, stride=2),
            EncoderBasic3DBlock(in_channels=16, out_channels=32, stride=2),
            EncoderBasic3DBlock(in_channels=32, out_channels=64, stride=2),
            nn.AdaptiveAvgPool3d((1, 8, 8)),
            nn.Flatten(start_dim=1, end_dim=2),
            EncoderBasic2DBlock(in_channels=64, out_channels=128, stride=2),  # b,128,4,4
        )
        self.encoder2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 2048),
            nn.ReLU(),

            nn.Unflatten(1, (32, 4, 4, 4)),

            DecoderBasic3DBlock(in_channels=32, out_channels=32, use_pixel_shuffle=True),
            DecoderBasic3DBlock(in_channels=32, out_channels=32, use_pixel_shuffle=True),
            DecoderBasic3DBlock(in_channels=32, out_channels=64, use_pixel_shuffle=False,
                                upsample_shape=(20, 32, 32)),
            DecoderBasic3DBlock(in_channels=64, out_channels=64, use_pixel_shuffle=False,
                                upsample_shape=(20, 64, 64)),

            nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        self.last_conv = self.encoder1(x)
        h = self.encoder2(self.last_conv)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def forward(self, x):
        self.image_size = x.size()[-1]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # att_loss = self.attention_loss(z)
        return self.decode(z), mu, logvar

    def gen_from_noise(self, z):
        return self.decode(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        # torch.norm(tensor, dim=(2, 3), keepdim=True)
        return tensor / torch.norm(tensor)

    def _get_grad_weights(self, grad_z: Tensor) -> Tensor:
        """Applies the GAP operation to the gradients to obtain weights alpha."""

        alpha = self.normalize(grad_z)
        alpha = F.avg_pool2d(grad_z, kernel_size=grad_z.size(-1))
        return alpha

    def get_att_maps(self, z):
        grad_z = torch.autograd.grad(z, self.last_conv,
                                     grad_outputs=torch.ones_like(z), only_inputs=True)[0]

        alpha = self._get_grad_weights(grad_z)
        maps = F.relu(alpha * grad_z)
        maps = maps.unsqueeze(2)
        maps = F.interpolate(
            maps,
            size=(20, self.image_size, self.image_size),
            mode='trilinear',
            align_corners=False)
        return maps

    def attention_loss(self, z):
        maps = self.get_att_maps(z)
        a = torch.sum(min(maps, dim=1))
        return


if __name__ == '__main__':

    model = Conv3dVAE(latent_dim=32)

    x = torch.randn(32, 1, 20, 64, 64)

    a, mu, logvar = model(x)

    print(a.shape)
