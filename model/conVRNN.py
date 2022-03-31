from collections import OrderedDict
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.base_model import BaseModel
from model.block2d import Decoder2DBlock, Encoder2DBlock
from model.temporal_conv import TemporalConv1D, TemporalConv3D
from model.vae_loss import VRNNLoss, kld_gauss


class ConVRNN(BaseModel):
    def __init__(self, h_dim: int, latent_dim: int, activation: str,
                 n_layers: int = 1, bias: bool = False, bidirectional: bool = False):
        super(ConVRNN, self).__init__()

        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.name = "conVRNN"
        self.mask = torch.from_numpy(np.load(os.path.join(base_path, 'mask.npy')))

        act: nn.Module = {'relu': nn.ReLU(inplace=True),
                          'leakyrelu': nn.LeakyReLU(inplace=True),
                          'elu': nn.ELU(inplace=True),
                          'silu': nn.SiLU(inplace=True)}[activation]

        # feature-extracting transformations
        # input_dim -> hidden_dim
        # channels not changed
        self.phi_x = nn.Sequential(
            Encoder2DBlock(1, 16, stride=2, activation=act),  # 32x32
            Encoder2DBlock(16, 32, stride=2, activation=act),  # 16x16
            Encoder2DBlock(32, 32, stride=2, activation=act),  # 8x8
            # Encoder2DBlock(32, 64, stride=2, activation=act),  # 4x4
            # nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.temporal_conv = nn.Sequential(
            #TemporalConv1D(512, 512, activation=act, masked=True),
            TemporalConv1D(32 * 8 * 8, h_dim, activation=act, masked=True),
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

        # self.dec_std = nn.Sequential(
        #     nn.Unflatten(1, (32, 2, 2)),
        #     Decoder2DBlock(32, 32, upscale_factor=2, activation=act),  # 8x8
        #     Decoder2DBlock(32, 16, upscale_factor=2, activation=act),  # 8x8
        #     Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 16x16
        #     # Decoder2DBlock(16, 16, upscale_factor=2, activation=act),  # 16x16
        #     nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
        #     nn.Softplus()
        # )

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

        self.h_init = nn.Parameter(torch.zeros(n_layers, 1, h_dim))

    def encode(self, x: Tensor) -> Tensor:
        """Performs the encode pipeline on the input x.

        Args:
            x (Tensor): shape (seq_len, batch, channels, h, w)
        Returns:
            Tensor: shape (t, b, h_dim)
        """
        t, b, ch, h, w = x.size()
        x = x.reshape(t * b, ch, h, w)
        x = self.phi_x(x)
        if len(x.shape) == 2:  # already flattened
            x = x.view(t, b, -1)  # (t, b, h_dim)
            x = x.permute(1, 2, 0)  # (b, h_dim, t) # for 1d conv
            x = self.temporal_conv(x)
            x = x.permute(2, 0, 1)  # (t, b, h_dim)
            return x
        elif len(x.shape) == 4:  # not flattened
            _, ch, h, w = x.size()
            x = x.view(t, b, ch, h, w)
            x = x.permute(1, 2, 0, 3, 4)  # (b, ch, t, h, w) for 3d conv
            x = self.temporal_conv(x)
            x = x.permute(2, 0, 1, 3, 4)  # (t, b, ch, h, w)
            x = x.reshape(t, b, -1).contiguous()
            return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:

        x = x.permute(2, 0, 1, 3, 4)  # t, b, ch, h, w
        phi_x = self.encode(x)  # t, b, h
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        all_prior_mean, all_prior_std = [], []

        x_recon = torch.zeros(x.shape).to(self._device)
        h = self.h_init.expand(self.n_layers, x.shape[1], self.h_dim).contiguous()

        for t in range(x.size(0)):

            phi_x_t = phi_x[t]

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
            # dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            # all_dec_std.append(dec_std_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)

            x_recon[t] = dec_mean_t

        all_enc_mean = torch.stack(all_enc_mean, dim=0)
        all_enc_std = torch.stack(all_enc_std, dim=0)
        all_dec_mean = torch.stack(all_dec_mean, dim=0)
        # all_dec_std = torch.stack(all_dec_std, dim=0)
        all_prior_mean = torch.stack(all_prior_mean, dim=0)
        all_prior_std = torch.stack(all_prior_std, dim=0)

        x_recon = x_recon.permute(1, 2, 0, 3, 4)  # b, ch, t, h, w
        return x_recon, (all_enc_mean, all_enc_std), (all_prior_mean, all_prior_std)

    def _set_hook_func(self) -> None:
        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.named_modules():
            module[1].register_forward_hook(func_f)

    def _get_conv_outputs(self, outputs: Tensor,
                          target_layer: Tensor) -> Tensor:
        for i, (k, v) in enumerate(outputs.items()):
            for module in self.named_modules():
                if k == id(module[1]) and module[0] == target_layer:
                    return v

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        # torch.norm(tensor, dim=(2, 3), keepdim=True)
        return tensor / torch.linalg.norm(tensor, dim=(2, 3), keepdim=True)

    def _get_grad_weights(self, grad_z: Tensor) -> Tensor:
        """Applies the GAP operation to the gradients to obtain weights alpha."""

        alpha = self.normalize(grad_z)
        alpha = F.avg_pool2d(grad_z, kernel_size=grad_z.size(-1))
        return alpha

    def attention_maps(self, x: Tensor, gradcam_pp: bool = 'False') -> Tuple[Tensor, Tensor]:
        self.outputs_forward = OrderedDict()
        self._set_hook_func()
        batch, ch, seq, height, width = x.shape
        x = x.permute(2, 0, 1, 3, 4)  # t, b, ch, h, w

        x_recon = torch.zeros_like(x)
        maps = torch.zeros((seq, *x_recon.shape))
        h = self.h_init.expand(self.n_layers, x.shape[1], self.h_dim).contiguous()

        old_conv_outputs = []

        for t in range(x.size(0)):

            # questo phi_x_t è diverso dal farlo con tutti insieme
            phi_x_t = self.encode(x[t].unsqueeze(0)).squeeze()

            conv_outputs = self._get_conv_outputs(self.outputs_forward, target_layer='phi_x.2')
            old_conv_outputs.append(conv_outputs)

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
            # dec_std_t = self.dec_std(dec_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            x_recon[t] = dec_mean_t.detach()
            # mse = F.mse_loss(dec_mean_t, x[t], reduction='none')

            # kld = kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            # mask = torch.stack([self.mask.reshape((ch, height, width))] * batch).to(self._device)

            for c, conv in enumerate(old_conv_outputs):
                grad = torch.autograd.grad(
                    outputs=z_t, inputs=conv,
                    grad_outputs=torch.ones_like(z_t), retain_graph=True, only_inputs=True)[0]

                maps[c, t] = self.get_attention_map(height, width, conv, grad, gradcam_pp=gradcam_pp)

        return x_recon.permute(1, 2, 0, 3, 4), maps.permute(2, 1, 3, 0, 4, 5)

    def get_attention_map(self, height, width, conv_outputs, grad_z, gradcam_pp: bool = False):

        if not gradcam_pp:
            # alpha = self._get_grad_weights(grad_z)

            # maps_t = F.relu(torch.sum((alpha * conv_outputs), dim=1)).unsqueeze(1)
            weights = torch.mean(grad_z, dim=(2, 3))
        elif gradcam_pp:
            grads_power_2 = grad_z**2
            grads_power_3 = grads_power_2 * grad_z
            # Equation 19 in https://arxiv.org/abs/1710.11063
            sum_activations = torch.sum(conv_outputs, dim=(2, 3))
            eps = 0.000001
            aij = grads_power_2 / (2 * grads_power_2 +
                                   sum_activations[:, :, None, None] * grads_power_3 + eps)
            # Now bring back the ReLU from eq.7 in the paper,
            # And zero out aijs where the activations are 0
            aij = torch.where(grad_z != 0, aij, torch.tensor(0.0, device=self._device))

            weights = F.relu(grad_z) * aij
            weights = torch.sum(weights, dim=(2, 3))

        maps_t = F.relu((weights[:, :, None, None] * conv_outputs).sum(dim=1, keepdim=True))
        maps_t = F.interpolate(
            maps_t,
            size=(height, width),
            mode='bilinear',
            align_corners=False)

        return maps_t

    @staticmethod
    def get_loss_function(**kwargs) -> nn.Module:
        return VRNNLoss()

    def sample(self, seq_len: int = 20) -> Tensor:

        sequence = []
        h = self.h_init.expand(self.n_layers, 1, self.h_dim).contiguous()

        for t in range(seq_len):
            if t == 0:
                # prior
                prior_t = self.prior(h[-1])
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                z_t = self.reparameterize(prior_mean_t, prior_std_t)
            else:
                # encoder
                enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                z_t = self.reparameterize(enc_mean_t, enc_std_t)

            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            # dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)
            phi_x_t = torch.flatten(phi_x_t, start_dim=1)  # when not already flattened, otherwise is equality
            # in train the temporal conv casts to h_dim, here we need to use pooling
            # when it is already the same dimension, it doesnt change
            phi_x_t = F.adaptive_avg_pool1d(phi_x_t.unsqueeze(0), self.h_dim).squeeze(0)

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
    x = torch.randn(16, 1, 20, 64, 64, device=device)

    rec = model.attention_maps(x)
    print(rec.shape)
