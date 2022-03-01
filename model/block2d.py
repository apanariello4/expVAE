import torch.nn as nn


class Encoder2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation: nn.Module = nn.ReLU(inplace=True)):
        super(Encoder2DBlock, self).__init__()
        kernel_size = 3
        padding = 1
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.downsample(residual))


class Decoder2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation: nn.Module = nn.ReLU(inplace=True)):
        super(Decoder2DBlock, self).__init__()
        kernel_size = 3
        padding = 1

        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.upsample(residual))
