import torch
from torch.nn import functional as F
from torch import Tensor


def vae_loss(x_recon: Tensor, x: Tensor, mu: Tensor,
             logvar: Tensor, alpha: float = 1.0) -> Tensor:

    reconstruction_error = F.binary_cross_entropy(x_recon, x, reduction='sum')

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_error + alpha * kld
