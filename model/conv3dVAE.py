from typing import Tuple
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F
from utils.pixel_shuffle import PixelShuffle3d


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

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 8, 8)),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.SiLU()
        )

        # flatten: (1,128,7,7) -> (1,128*7*7) = (1,6272)
        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.SiLU(),

            nn.Linear(1024, 2048),
            nn.SiLU(),

            nn.Unflatten(1, (32, 4, 4, 4)),

            nn.Conv3d(32, 32 * 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            PixelShuffle3d(2),

            nn.Conv3d(32, 32 * 8, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            PixelShuffle3d(2),

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample((20, 32, 32), mode='trilinear'),

            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Upsample((20, 64, 64), mode='trilinear'),

            nn.Conv3d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

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


if __name__ == '__main__':
    # model = Conv3dVAE(latent_dim=32)

    # x = torch.randn(32, 1, 20, 64, 64)

    # a = nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1)(x)
    # a = nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1)(a)
    # a = nn.Conv3d(32, 64, kernel_size=4, stride=(1, 2, 2), padding=1)(a)
    # a = nn.AdaptiveAvgPool3d((1, 8, 8))(a)
    # a = nn.Flatten(start_dim=1, end_dim=2)(a)
    # a = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)(a)
    # a = nn.Flatten()(a)
    # a = nn.Linear(2048, 1024)(a)

    x = torch.randn(32, 32)

    a = nn.Linear(32, 1024)(x)
    a = nn.Linear(1024, 2048)(a)
    a = nn.Unflatten(1, (128, 4, 4))(a)
    a = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)(a)
    a = nn.Unflatten(1, (64, 1))(a)
    a = nn.Upsample((4, 8, 8), mode='trilinear')(a)
    a = nn.ConvTranspose3d(
        64, 32, kernel_size=4, stride=(
            1, 2, 2), padding=1)(a)
    a = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)(a)
    a = nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1)(a)

    print(a.shape)
