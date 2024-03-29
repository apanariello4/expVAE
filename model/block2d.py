import torch.nn as nn
from torch import Tensor


class Encoder2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 activation: nn.Module = nn.ReLU(inplace=True), bias: bool = False):
        super(Encoder2DBlock, self).__init__()
        kernel_size = 3
        padding = 1
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.downsample(residual))


class Decoder2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upscale_factor: int = 2,
                 activation: nn.Module = nn.ReLU(inplace=True), bias: bool = False):
        super(Decoder2DBlock, self).__init__()

        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.ups1 = nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ups_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Upsample(scale_factor=upscale_factor, mode='bilinear', align_corners=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.conv1(x)
        x = self.ups1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.ups_res(residual))


if __name__ == '__main__':
    import torch
    x = torch.randn(320, 64, 16, 16)

    encoder = Decoder2DBlock(64, 32)

    y = encoder(x)
    print(y.shape)
