from typing import Tuple
import torch.nn as nn
import torch
from torch import Tensor


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.name = 'convVAE'

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # flatten: (1,128,7,7) -> (1,128*7*7) = (1,6272)
        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_dim)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
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
