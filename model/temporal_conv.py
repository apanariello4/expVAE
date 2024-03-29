import torch.nn as nn
import numpy as np


class TemporalConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 activation: nn.Module = nn.ReLU(inplace=True), masked: bool = False, bias: bool = False):
        super(TemporalConv1D, self).__init__()

        conv1d = MaskedConv1D if masked else nn.Conv1d

        self.activation = activation
        self.conv1 = conv1d(in_channels, out_channels,
                            kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = conv1d(out_channels, out_channels,
                            kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = conv1d(in_channels, out_channels,
                                 kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.downsample(residual))


class MaskedConv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv1D, self).__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, k = self.weight.size()
        self.mask.fill_(0)

        self.mask[:, :, :np.ceil(k / 2).astype(int)] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1D, self).forward(x)


class TemporalConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(TemporalConv3D, self).__init__()

        self.activation = activation
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0))
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = nn.Conv3d(in_channels, out_channels,
                                    kernel_size=(3, 1, 1), stride=stride, padding=(1, 0, 0))

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.downsample(residual))


if __name__ == '__main__':
    import torch
    #x = torch.randn(16, 20, 32, 8, 8)
    #model = TemporalConv_3d(32, 32, 1)
    x = torch.randn(1, 512, 1)
    model = TemporalConv1D(512, 512, masked=True)
    y = model(x)
    print(y.shape)
