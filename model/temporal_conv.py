import torch.nn as nn


class TemporalConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(TemporalConv1D, self).__init__()

        self.activation = activation
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.activation(x + self.downsample(residual))


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
    x = torch.randn(16, 20, 512)
    model = TemporalConv1D(20, 20, 1)
    y = model(x)
    print(y.shape)
