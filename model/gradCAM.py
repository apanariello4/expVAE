from collections import OrderedDict
from importlib.metadata import distribution
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from model.base_model import BaseModel


class GradCAM():
    def __init__(self, model: BaseModel):
        self.model = model

        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self._set_hook_func()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_hat, *distribution = self.model(x)
        self.image_size = x.size(-1)
        return x_hat, *distribution[0]

    def _set_hook_func(self) -> None:
        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)

    def _get_conv_outputs(self, outputs: Tensor,
                          target_layer: Tensor) -> Tensor:
        for i, (k, v) in enumerate(outputs.items()):
            for module in self.model.named_modules():
                if k == id(module[1]) and module[0] == target_layer:
                    return v

    @staticmethod
    def normalize(tensor: Tensor) -> Tensor:
        # torch.norm(tensor, dim=(2, 3), keepdim=True)
        return tensor / torch.norm(tensor)

    def _get_grad_weights(self, grad_z: Tensor) -> Tensor:
        """Applies the GAP operation to the gradients to obtain weights alpha."""

        alpha = self.normalize(grad_z)
        alpha = F.avg_pool2d(grad_z, kernel_size=grad_z.size(-1))
        return alpha

    def get_attention_map(self, x: Tensor, x_hat: Tensor, mu: Tensor, logvar: Tensor,
                          target_layer: str = 'encoder.2') -> Tensor:
        """Generates attention map from gradients."""

        z = self.model.reparameterize(mu, logvar)

        reconstruction_loss = F.binary_cross_entropy(
            x_hat, x, reduction='none')

        conv_output = self._get_conv_outputs(
            self.outputs_forward, target_layer)

        # backprop z wrt to conv_output
        grad_z = torch.autograd.grad(
            outputs=z, inputs=conv_output,
            grad_outputs=torch.ones(z.size()).to(self.model._device), retain_graph=True,
            only_inputs=True)[0]

        grad_loss = torch.autograd.grad(
            outputs=reconstruction_loss, inputs=conv_output,
            grad_outputs=torch.ones(
                reconstruction_loss.size()).to(
                self.model._device),
            only_inputs=True)[0]

        self.model.zero_grad()

        alpha = self._get_grad_weights(grad_z)
        beta = self._get_grad_weights(grad_loss)

        maps = F.relu(torch.sum((alpha * conv_output), dim=1)).unsqueeze(1)
        maps = F.interpolate(
            maps,
            size=self.image_size,
            mode='bilinear',
            align_corners=False)

        # maps_loss = F.relu(beta * grad_loss)
        # maps_loss = F.interpolate(
        #     maps_loss,
        #     size=self.image_size,
        #     mode='bilinear',
        #     align_corners=False)

        return maps  # , maps_loss
