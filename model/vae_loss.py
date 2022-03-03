from typing import Optional, Tuple, Union
import torch
from torch.nn import functional as F
from torch import Tensor


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

    if return_recon_kld:
        return loss, reconstruction_error, kld

    return loss


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
