from typing import Optional, Tuple, Union
import torch
from torch.nn import functional as F
from torch import Tensor


def vae_loss(x_recon: Tensor, x: Tensor, mu: Tensor,
             logvar: Tensor, alpha: float = 1.0, return_min_max: bool = False):
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
    reconstruction_error = F.binary_cross_entropy(x_recon, x, reduction='sum')

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = reconstruction_error + alpha * kld

    if return_min_max:
        sample_recon_err = F.binary_cross_entropy(x_recon, x, reduction='none').squeeze()
        batch_size, depth = sample_recon_err.shape[0], sample_recon_err.shape[1]
        sample_recon_err = sample_recon_err.view(batch_size, depth, -1).sum(-1)

        sample_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        min_max_loss = (sample_recon_err.min(), sample_recon_err.max(),
                        sample_kld.min(), sample_kld.max())

        return loss, min_max_loss

    return loss


def min_max_normalization(x: Tensor, new_min: float, new_max: float) -> Tensor:
    return (x - new_min) / (new_max - new_min)


def vae_loss_normalized(x_recon: Tensor, x: Tensor, mu: Tensor,
                        logvar: Tensor, min_max: Tuple[float], alpha: float = 1.0) -> Tensor:

    sample_recon_err = F.binary_cross_entropy(x_recon, x, reduction='none').squeeze()
    batch_size, depth = sample_recon_err.shape[0], sample_recon_err.shape[1]
    sample_recon_err = sample_recon_err.view(batch_size, depth, -1).sum(-1)
    recon_err_normalized = min_max_normalization(sample_recon_err, *min_max[:2])

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kld_normalized = min_max_normalization(kld, *min_max[2:])

    loss = recon_err_normalized.sum() + alpha * kld_normalized.sum()

    return loss, recon_err_normalized, kld_normalized
