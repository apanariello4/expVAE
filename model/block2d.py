import torch.nn as nn


class Encoder2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Encoder2DBlock, self).__init__()
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.silu = nn.SiLU()

        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.silu(x + self.downsample(residual))


class Decoder2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Decoder2DBlock, self).__init__()
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.silu(x + self.upsample(residual))
