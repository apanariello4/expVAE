import torch.nn as nn
from model.pixel_shuffle import PixelShuffle3d


class Encoder3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_bias=False,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(Encoder3DBlock, self).__init__()

        self.act = activation

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=use_bias),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return self.act(x + self.downsample(residual))


class Decoder3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=False,
                 use_pixel_shuffle=False, upsample_shape: tuple = None,
                 activation: nn.Module = nn.ReLU(inplace=True)):

        super(Decoder3DBlock, self).__init__()
        assert upsample_shape or use_pixel_shuffle, 'Either upsample_shape or use_pixel_shuffle must be specified'

        if use_pixel_shuffle:
            assert upsample_shape is None

        if upsample_shape is not None:
            assert use_pixel_shuffle is False and isinstance(upsample_shape, tuple) \
                and len(upsample_shape) == 3

        self.ups = use_pixel_shuffle

        in1, out1 = in_channels, out_channels * (2**3) if self.ups else out_channels
        in2, out2 = out_channels, out_channels

        self.conv1 = nn.Conv3d(in1, out1, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn1 = nn.BatchNorm3d(in2)

        self.conv2 = nn.Conv3d(in2, out2, kernel_size=3,
                               stride=1, padding=1, bias=use_bias)
        self.bn2 = nn.BatchNorm3d(out2)

        self.act = activation

        self.conv3 = nn.Conv3d(in1, out1, kernel_size=1,
                               stride=1, padding=0, bias=use_bias)
        self.bn3 = nn.BatchNorm3d(in2)

        self.upsample = PixelShuffle3d(2) if self.ups else \
            nn.Upsample(upsample_shape, mode='trilinear', align_corners=False)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.upsample(x)
        x = self.act(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        residual = self.conv3(residual)
        residual = self.upsample(residual)
        residual = self.bn3(residual)

        out = self.act(x + residual)

        return out
