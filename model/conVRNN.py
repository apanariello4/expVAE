import torch.nn as nn
import torch
from torch import Tensor
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.block2d import Decoder2DBlock, Encoder2DBlock
from model.vae_loss import VRNNLoss
from model.base_model import BaseModel


EPS = torch.finfo(torch.float).eps


class ConVRNN(BaseModel):
    def __init__(self, h_dim: int, latent_dim: int, activation: str,
                 n_layers: int = 1, bias: bool = False, bidirectional: bool = False):
        super(ConVRNN, self).__init__()

        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.name = "conVRNN"

        act = {'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               'elu': nn.ELU(inplace=True),
               'silu': nn.SiLU(inplace=True)}[activation]

        # feature-extracting transformations
        # input_dim -> hidden_dim
        # channels not changed
        self.phi_x = nn.Sequential(
            Encoder2DBlock(1, 16, stride=4, activation=act),  # 32x32
            Encoder2DBlock(16, 32, stride=2, activation=act),  # 16x16
            Encoder2DBlock(32, 32, stride=2, activation=act),  # 8x8
            # Encoder2DBlock(64, 128, stride=2, activation=act),  # 4x4
            # nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU())

        # encoder

        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )

        self.enc_mean = nn.Linear(h_dim, latent_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, latent_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, latent_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, latent_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        self.dec_std = nn.Sequential(
            nn.Unflatten(1, (32, 2, 2)),
            Decoder2DBlock(32, 32, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(32, 16, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 16x16
            # Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 16x16
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus()
        )

        self.dec_mean = nn.Sequential(
            nn.Unflatten(1, (32, 4, 4)),
            Decoder2DBlock(32, 32, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(32, 16, upscale_factor=2, activation=act),  # 16x16
            Decoder2DBlock(16, 16, upscale_factor=4, activation=act),  # 32x32
            # Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 64x64
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.rnn = nn.GRU(input_size=h_dim + h_dim, hidden_size=h_dim,
                          num_layers=n_layers, bias=bias)

        # if self.bidirectional:
        #     self.reverse_rnn = nn.GRU(input_size=h_dim + h_dim, hidden_size=h_dim)
        #     self.reverse_rnn.weight_ih_l0 = self.rnn.weight_ih_l0_reverse
        #     self.reverse_rnn.weight_hh_l0 = self.rnn.weight_hh_l0_reverse
        #     if bias:
        #         self.reverse_rnn.bias_ih_l0 = self.rnn.bias_ih_l0_reverse
        #         self.reverse_rnn.bias_hh_l0 = self.rnn.bias_hh_l0_reverse

        self.h_init = nn.Parameter(torch.zeros(n_layers, 1, h_dim))

    def forward(self, x: Tensor):

        x = x.permute(2, 0, 1, 3, 4)  # t, b, ch, h, w

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        kld_loss = 0
        nll_loss = 0
        self.sample_dim = x.shape[-3:]

        x_recon = torch.zeros(x.shape).to(self._device)
        h = self.h_init.expand(self.n_layers, x.shape[1], self.h_dim).contiguous()

        # h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=self._device)
        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            # computing losses
            # kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)

            x_recon[t] = dec_mean_t

        all_enc_mean = torch.stack(all_enc_mean, dim=0)
        all_enc_std = torch.stack(all_enc_std, dim=0)
        all_dec_mean = torch.stack(all_dec_mean, dim=0)
        all_dec_std = torch.stack(all_dec_std, dim=0)
        all_prior_mean = torch.stack(all_prior_mean, dim=0)
        all_prior_std = torch.stack(all_prior_std, dim=0)

        x_recon = x_recon.permute(1, 2, 0, 3, 4)  # b, ch, t, h, w
        return x_recon, (all_enc_mean, all_enc_std), (all_prior_mean, all_prior_std)

    @staticmethod
    def get_loss_function(**kwargs) -> nn.Module:
        return VRNNLoss()

    def sample(self, seq_len: int = 20) -> Tensor:

        sequence = []
        h = self.h_init.expand(self.n_layers, 1, self.h_dim).contiguous()

        for _ in range(seq_len):

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self.reparameterize(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sequence.append(dec_mean_t.data)

        return torch.stack(sequence, dim=0).squeeze(1)

    @ property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConVRNN(h_dim=512, latent_dim=512, activation='elu', bidirectional=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # model.load_state_dict(torch.load('/home/nello/expVAE/checkpoints/h_0-sample__conVRNN_03071431best.pth')['state_dict'])
    model.to(device)
    x = torch.randn(12, 1, 20, 64, 64, device=device)

    y = torch.randn(128, 4, 4)
    y = nn.AvgPool2d((4, 4))(y)

    rec = model(x)
    print(rec[0].shape)
