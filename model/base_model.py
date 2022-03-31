import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(nn.Module):
    @property
    def count_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def _device(self) -> torch.device:
        """Returns the device of the model."""
        return next(self.parameters()).device

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_loss_function():
        raise NotImplementedError

    def sample() -> Tensor:
        raise NotImplementedError
