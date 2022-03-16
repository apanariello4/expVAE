from unicodedata import bidirectional
import torch.nn as nn
import torch
from torch import Tensor
import os
import sys

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.block2d import Decoder2DBlock, Encoder2DBlock
from model.vae_loss import BIVRNNLoss
from model.base_model import BaseModel


EPS = torch.finfo(torch.float).eps


class BidirectionalConVRNN(BaseModel):
    def __init__(self, h_dim: int, latent_dim: int, activation: str,
                 n_layers: int = 1, bias: bool = False):
        super(BidirectionalConVRNN, self).__init__()

        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.name = "bi-conVRNN"

        act = {'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               'elu': nn.ELU(inplace=True),
               'silu': nn.SiLU(inplace=True)}[activation]

        # feature-extracting transformations
        # input_dim -> hidden_dim
        # channels not changed
        self.phi_x = nn.Sequential(
            Encoder2DBlock(1, 16, stride=2, activation=act),  # 32x32
            Encoder2DBlock(16, 32, stride=2, activation=act),  # 16x16
            Encoder2DBlock(32, 64, stride=2, activation=act),  # 8x8
            Encoder2DBlock(64, 128, stride=2, activation=act),  # 4x4
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        self.phi_z = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU()
        )

        self.h_tilde_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )
        self.h_tilde_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        self.b_tilde_mean = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )
        self.b_tilde_std = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Softplus()
        )

        self.b_tilde = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LeakyReLU(),
        )

        # encoder

        self.enc = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )

        self.enc_mean = nn.Linear(h_dim, latent_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, latent_dim),
            nn.Softplus())

        # prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU())
        self.prior_mean = nn.Linear(h_dim, latent_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, latent_dim),
            nn.Softplus())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )

        self.dec_std = nn.Sequential(
            nn.Unflatten(1, (128, 2, 2)),
            Decoder2DBlock(128, 64, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(64, 32, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(32, 16, upscale_factor=2, activation=act),  # 16x16
            Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 32x32
            Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 64x64
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

        self.dec_b = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU()
        )
        self.dec_mean_b = nn.Sequential(
            nn.Unflatten(1, (32, 4, 4)),
            Decoder2DBlock(32, 32, upscale_factor=2, activation=act),  # 8x8
            Decoder2DBlock(32, 16, upscale_factor=2, activation=act),  # 16x16
            Decoder2DBlock(16, 16, upscale_factor=4, activation=act),  # 32x32
            # Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 64x64
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.rnn = nn.GRU(input_size=h_dim * 3, hidden_size=h_dim,
                          num_layers=n_layers, bias=bias, bidirectional=False)

        self.back_rnn = nn.GRU(input_size=h_dim, hidden_size=h_dim,
                               num_layers=n_layers, bias=bias, bidirectional=False)

        self.rnn_loss = BIVRNNLoss()

        self.h_init = nn.Parameter(torch.zeros(n_layers, 1, h_dim))
        self.b_init = nn.Parameter(torch.zeros(n_layers, 1, h_dim))

    def encode(self, x):
        t, b, ch, h, w = x.size()
        x = x.reshape(t * b, ch, h, w)
        x = self.phi_x(x)
        if len(x.shape) == 2:
            return x.view(t, b, -1)
        assert len(x.shape) == 5
        _, ch, h, w = x.size()
        return x.view(t, b, ch, h, w)

    def forward(self, x: Tensor):

        x = x.permute(2, 0, 1, 3, 4)  # b, ch, t, h, w -> t, b, ch, h, w
        phi_x = self.encode(x)

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_b_tilde_mean, all_b_tilde_std = [], []
        all_h_tilde_mean, all_h_tilde_std = [], []
        all_b, all_h = [], []

        x_recon = torch.zeros(x.shape).to(self._device)
        x_recon_b = torch.zeros(x.shape).to(self._device)
        h = self.h_init.expand(self.n_layers, x.shape[1], self.h_dim).contiguous()
        b = self.b_init.expand(self.n_layers, x.shape[1], self.h_dim).contiguous()

        for t in range(x.size(0)):

            phi_x_t = phi_x[t]

            # encoder
            enc_t = self.enc(torch.cat([h[-1], b[-1]], 1))

            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self.reparameterize(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            b_tilde_mean = self.b_tilde_mean(phi_z_t)
            b_tilde_std = self.b_tilde_std(phi_z_t)
            b_tilde = self.b_tilde(torch.cat([b_tilde_mean, b_tilde_std], 1))

            h_tilde_mean = self.h_tilde_mean(phi_z_t)
            h_tilde_std = self.h_tilde_std(phi_z_t)

            # input_to_rnn = torch.cat([phi_z_t, b_tilde, phi_x_t], 1)

            # # add sequence dimension
            # input_to_rnn = input_to_rnn.unsqueeze(0)

            # global_state = torch.cat((h, b), 0)
            # # recurrence
            # _, new_global_state = self.rnn(input_to_rnn, global_state)
            # b_t = b
            # h, b = new_global_state.chunk(2, 0)

            input_to_rnn = torch.cat([phi_z_t, b_tilde, phi_x_t], 1).unsqueeze(0)
            _, h = self.rnn(input_to_rnn, h)

            # decoder
            dec_t = self.dec(h[-1])
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            dec_t_b = self.dec_b(b[-1])
            dec_mean_t_b = self.dec_mean_b(dec_t_b)

            _, b = self.back_rnn(phi_x_t.unsqueeze(0), b)

            # computing losses
            # kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_b_tilde_mean.append(b_tilde_mean)
            all_b_tilde_std.append(b_tilde_std)
            all_h_tilde_mean.append(h_tilde_mean)
            all_h_tilde_std.append(h_tilde_std)
            all_b.append(b.squeeze())
            all_h.append(h.squeeze())

            x_recon[t] = dec_mean_t
            x_recon_b[t] = dec_mean_t_b

        all_enc_mean = torch.stack(all_enc_mean, dim=0)
        all_enc_std = torch.stack(all_enc_std, dim=0)
        all_dec_mean = torch.stack(all_dec_mean, dim=0)
        all_dec_std = torch.stack(all_dec_std, dim=0)
        all_prior_mean = torch.stack(all_prior_mean, dim=0)
        all_prior_std = torch.stack(all_prior_std, dim=0)
        all_b_tilde_mean = torch.stack(all_b_tilde_mean, dim=0)
        all_b_tilde_std = torch.stack(all_b_tilde_std, dim=0)
        all_h_tilde_mean = torch.stack(all_h_tilde_mean, dim=0)
        all_h_tilde_std = torch.stack(all_h_tilde_std, dim=0)
        all_b = torch.stack(all_b, dim=0)
        all_h = torch.stack(all_h, dim=0)

        x_recon = x_recon.permute(1, 2, 0, 3, 4)  # t, b, ch, h, w -> b, ch, t, h, w
        x_recon_b = x_recon_b.permute(1, 2, 0, 3, 4)  # t, b, ch, h, w -> b, ch, t, h, w
        return (x_recon, x_recon_b), (all_enc_mean, all_enc_std), (all_prior_mean, all_prior_std),\
            (all_b_tilde_mean, all_b_tilde_std), (all_h_tilde_mean, all_h_tilde_std), (all_b, all_h)

    @ staticmethod
    def get_loss_function(**kwargs) -> nn.Module:
        return BIVRNNLoss()

    def sample(self, seq_len: int = 20) -> Tensor:

        return torch.zeros((20, 1, 64, 64), device=self._device)

        sequence = []
        # h = torch.zeros(self.n_layers, 1, self.h_dim, device=self._device)
        # h = self.h_init.expand(self.n_layers, 1, self.h_dim).contiguous()
        h = torch.randn(self.n_layers, 1, self.h_dim, device=self._device)

        for t in range(seq_len):

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self.reparameterize(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            b_tilde_mean = self.b_tilde_mean(phi_z_t)
            b_tilde_std = self.b_tilde_std(phi_z_t)

            input_to_rnn = torch.cat([phi_z_t, b_tilde_mean,
                                      b_tilde_std, phi_x_t], 1)

            # add sequence dimension
            input_to_rnn = input_to_rnn.unsqueeze(0)

            global_state = torch.cat((h, b_tilde_mean), 0)
            # recurrence
            _, new_global_state = self.rnn(input_to_rnn, global_state)

            h, _ = new_global_state.chunk(2, 0)

            # decoder
            dec_t = self.dec(h[-1])
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            sequence.append(dec_mean_t.data)

        return torch.stack(sequence, dim=0).squeeze(1)

    @ property
    def _device(self) -> torch.device:
        return next(self.parameters()).device


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BidirectionalConVRNN(h_dim=512, latent_dim=512, activation='elu')
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # model.load_state_dict(torch.load('/home/nello/expVAE/checkpoints/h_0-sample__conVRNN_03071431best.pth')['state_dict'])
    model.to(device)
    x = torch.randn(12, 1, 20, 64, 64, device=device)

    rec = model(x)
    print(rec[0].shape)
