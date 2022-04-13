from typing import Optional, Tuple, Union
import torch
from torch.nn import functional as F
from torch import Tensor
import torch.nn as nn
from einops import rearrange

EPS = torch.finfo(torch.float).eps


def vae_loss(x_recon: Tensor, x: Tensor, mu: Tensor,
             logvar: Tensor, recon_func: str, alpha: float = 1.0,
             return_min_max: bool = False, return_recon_kld: bool = False):
    """
    Variational Autoencoder loss function
    :param x_recon: reconstructed input
    :param x: original input
    :param mu: mean of the latent distribution
    :param logvar: log variance of the latent distribution
    :param alpha: weight of the KL divergence
    :param return_min_max: whether to return the min and max of the reconstruction loss
    :return: loss, and optionally the min and max of the reconstruction loss
    """
    assert recon_func in ['mse', 'bce'], 'recon_func must be either mse or bce'
    assert not (return_min_max and return_recon_kld), 'return_min_max and return_recon_kld cannot be both True'

    if recon_func == 'mse':
        recon_function = F.mse_loss
    elif recon_func == 'bce':
        recon_function = F.binary_cross_entropy

    reconstruction_error = recon_function(x_recon, x, reduction='sum')

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = reconstruction_error + alpha * kld

    if return_min_max:
        sample_recon_err = recon_function(x_recon, x, reduction='none').squeeze()
        batch_size, depth = sample_recon_err.shape[0], sample_recon_err.shape[1]
        sample_recon_err = sample_recon_err.view(batch_size, depth, -1).sum(-1)

        sample_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        min_max_loss = (sample_recon_err.min(), sample_recon_err.max(),
                        sample_kld.min(), sample_kld.max())

        return loss, min_max_loss

    return loss, reconstruction_error, kld


def min_max_normalization(x: Tensor, new_min: float, new_max: float) -> Tensor:
    return (x - new_min) / (new_max - new_min)


def vae_loss_normalized(x_recon: Tensor, x: Tensor, mu: Tensor,
                        logvar: Tensor, min_max: Tuple[float],
                        alpha: float = 1.0, recon_func: str = 'bce') -> Tensor:

    if recon_func == 'mse':
        recon_function = F.mse_loss
    elif recon_func == 'bce':
        recon_function = F.binary_cross_entropy

    sample_recon_err = recon_function(x_recon, x, reduction='none').squeeze()
    batch_size, depth = sample_recon_err.shape[0], sample_recon_err.shape[1]
    sample_recon_err = sample_recon_err.view(batch_size, depth, -1).sum(-1)
    recon_err_normalized = min_max_normalization(sample_recon_err, *min_max[:2])

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kld_normalized = min_max_normalization(kld, *min_max[2:])

    loss = recon_err_normalized.sum() + alpha * kld_normalized.sum()

    return loss, recon_err_normalized, kld_normalized


class VAELoss(nn.Module):
    def __init__(self, recon_func: str):
        super().__init__()

        assert recon_func in ['mse', 'bce'], 'recon_func must be either mse or bce'
        #assert not (return_min_max and return_recon_kld), 'return_min_max and return_recon_kld cannot be both True'

        self.recon_func = recon_func

    def forward(self, x_recon: Tensor, x: Tensor, mu: Tensor,
                logvar: Tensor, return_min_max: bool = False, alpha: float = 1.0) -> Tensor:

        return vae_loss(x_recon, x, mu, logvar, self.recon_func, alpha, return_min_max)


def kld_gauss(mean_1: Tensor, std_1: Tensor,
              mean_2: Tensor, std_2: Tensor, frame_level: bool = False) -> Tensor:
    """Using std to compute KLD"""

    kld_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
                   (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                   std_2.pow(2) - 1)
    return 0.5 * torch.sum(kld_element, dim=-1) if frame_level else 0.5 * torch.sum(kld_element)


def nll_bernoulli(theta: Tensor, x: Tensor, frame_level: bool = False, seq_level: bool = False) -> Tensor:
    """Using log-likelihood to compute NLL.

        If frame_level is True, then the loss is computed for each frame,
        and the shape should be (batch_size, seq_len, ch*h*w)
    """
    assert theta.dim() == x.dim(), 'theta and x must have the same dimension'
    assert not (frame_level and seq_level), 'frame_level and seq_level cannot be both True'

    if frame_level:
        if theta.dim() != 3:
            theta = rearrange(theta, 'b c t h w -> b t (c h w)')
            x = rearrange(x, 'b c t h w -> b t (c h w)')
        nll = x * torch.log(theta + EPS) + (1 - x) * torch.log(1 - theta - EPS)
        return - torch.sum(nll, dim=-1)
    elif seq_level:
        if theta.dim() != 2:
            theta = rearrange(theta, 'b ... -> b (...)')
            x = rearrange(x, 'b ... -> b (...)')
        nll = x * torch.log(theta + EPS) + (1 - x) * torch.log(1 - theta - EPS)
        return - torch.sum(nll, dim=-1)
    else:
        nll = x * torch.log(theta + EPS) + (1 - x) * torch.log(1 - theta - EPS)
        return - torch.sum(nll)


def nll_gauss(mean: Tensor, std: Tensor, x: Tensor, frame_level: bool = False) -> Tensor:

    loss = F.gaussian_nll_loss(mean, x, std, reduction='none')
    return loss.sum(dim=-1) if frame_level else loss.sum()


class VRNNLoss(nn.Module):
    def __init__(self,):
        super(VRNNLoss, self).__init__()

    def frame_level_loss(self, x_recon, x,
                         mu_posterior, std_posterior,
                         mu_prior, std_prior, min_max: Tuple[Tensor] = None,
                         return_min_max: bool = False,):

        assert not (return_min_max and min_max), 'return_min_max and min_max cannot be both True'

        batch_size, depth = x.shape[0], x.shape[2]
        x = x.transpose(1, 2).reshape(batch_size, depth, -1)
        x_recon = x_recon.transpose(1, 2).reshape(batch_size, depth, -1)

        nll_frame_level = nll_bernoulli(x_recon, x, frame_level=True)

        mu_posterior = mu_posterior.transpose(0, 1)
        std_posterior = std_posterior.transpose(0, 1)
        mu_prior = mu_prior.transpose(0, 1)
        std_prior = std_prior.transpose(0, 1)

        kld_frame_level = kld_gauss(mu_posterior, std_posterior,
                                    mu_prior, std_prior, frame_level=True)
        if return_min_max:
            return nll_frame_level, kld_frame_level

        if min_max:
            nll_frame_level = min_max_normalization(nll_frame_level, *min_max[:2])
            kld_frame_level = min_max_normalization(kld_frame_level, *min_max[2:])

        return nll_frame_level + kld_frame_level, nll_frame_level, kld_frame_level

    def forward(self, x_recon: Tensor, x: Tensor,
                mu_std_posterior: Tuple[Tensor, Tensor],
                mu_std_prior: Tuple[Tensor, Tensor],
                alpha: float = 1.0, min_max_train: Tuple[Tensor] = None,
                frame_level: bool = False, return_min_max: bool = False) -> Tensor:

        assert not (return_min_max and frame_level), 'return_min_max and frame_level cannot be both True'

        mu_posterior, std_posterior = mu_std_posterior
        mu_prior, std_prior = mu_std_prior

        kld = kld_gauss(mu_posterior, std_posterior, mu_prior, std_prior)
        nll = nll_bernoulli(x_recon, x)

        loss = nll + alpha * kld

        if frame_level:
            loss, recon_error, kld_error = self.frame_level_loss(x_recon, x, mu_posterior, std_posterior, mu_prior, std_prior, min_max=min_max_train)

            return loss, recon_error, kld_error

        if return_min_max:
            nll_frame, kld_frame = self.frame_level_loss(x_recon, x, mu_posterior, std_posterior, mu_prior, std_prior, return_min_max=True)
            return nll_frame, kld_frame

        return loss, nll, kld


class BIVRNNLoss(nn.Module):
    def __init__(self,):
        super(BIVRNNLoss, self).__init__()

    def frame_level_loss(self, x_recon, x, mu_posterior, std_posterior, mu_prior, std_prior):
        batch_size, depth = x.shape[0], x.shape[2]
        x = x.transpose(1, 2).reshape(batch_size, depth, -1)
        x_recon = x_recon.transpose(1, 2).reshape(batch_size, depth, -1)

        nll_frame_level = nll_bernoulli(x_recon, x, frame_level=True)

        mu_posterior = mu_posterior.transpose(0, 1)
        std_posterior = std_posterior.transpose(0, 1)
        mu_prior = mu_prior.transpose(0, 1)
        std_prior = std_prior.transpose(0, 1)

        kld_frame_level = kld_gauss(mu_posterior, std_posterior,
                                    mu_prior, std_prior, frame_level=True)

        min_max_loss = (nll_frame_level.min(), nll_frame_level.max(),
                        kld_frame_level.min(), kld_frame_level.max())

        return nll_frame_level + kld_frame_level, nll_frame_level, kld_frame_level

    def forward(self, x_recon_x_recon_b: Tuple[Tensor, Tensor], x: Tensor,
                mu_std_posterior: Tuple[Tensor, Tensor],
                mu_std_prior: Tuple[Tensor, Tensor],
                mu_std_b_tilde: Tuple[Tensor, Tensor],
                mu_std_h_tilde: Tuple[Tensor, Tensor],
                b_h: Tuple[Tensor, Tensor],
                alpha: float = 1.0, beta: float = 0.1,
                gamma: float = 0.1, frame_level: bool = False) -> Tensor:

        x_recon, x_recon_b = x_recon_x_recon_b
        mu_posterior, std_posterior = mu_std_posterior
        mu_prior, std_prior = mu_std_prior
        mu_b_tilde, std_b_tilde = mu_std_b_tilde
        mu_h_tilde, std_h_tilde = mu_std_h_tilde
        b, h = b_h

        kld = kld_gauss(mu_posterior, std_posterior, mu_prior, std_prior)
        nll = nll_bernoulli(x_recon[:-1], x[1:])
        nll_b = nll_bernoulli(x_recon_b[:-1], x[1:])
        nll_b_tilde = nll_gauss(mu_b_tilde, std_b_tilde, b)
        nll_h_tilde = nll_gauss(mu_h_tilde, std_h_tilde, h)
        mse_b_tilde = F.mse_loss(mu_b_tilde, b, reduction='sum')
        mse_h_tilde = F.mse_loss(mu_h_tilde, h, reduction='sum')

        loss = nll + nll_b + alpha * kld + beta * nll_b_tilde + gamma * nll_h_tilde

        if frame_level:
            loss, recon_error, kld_error = self.frame_level_loss(x_recon, x, mu_posterior,
                                                                 std_posterior, mu_prior, std_prior)

            return loss, recon_error, kld_error

        # if wandb.run:
        #     wandb.log({'nll': nll, 'kld': kld, 'mse_btilde': mse_b_tilde, 'mse_htilde': mse_h_tilde})

        return loss


if __name__ == '__main__':
    loss = VRNNLoss()
    x = torch.randn(16, 1, 20, 64, 64)
    x_recon = torch.randn(16, 1, 20, 64, 64)
    mu_posterior = torch.randn(20, 16, 512)
    std_posterior = torch.randn(20, 16, 512)
    mu_prior = torch.randn(20, 16, 512)
    std_prior = torch.randn(20, 16, 512)

    loss_val, min_max_loss = loss(x_recon, x, (mu_posterior, std_posterior), (mu_prior, std_prior), return_min_max=True)
