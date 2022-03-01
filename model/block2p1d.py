import torch.nn as nn
from torch import Tensor
# from model.pixel_shuffle import PixelShuffle3d


def conv1x3x3(in_planes: int, mid_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)


def conv3x1x1(mid_planes: int, planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Encoder2p1Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(Encoder2p1Block, self).__init__()

        self.activation = activation

        n_3d_parameters1 = in_channels * out_channels * 3 * 3 * 3
        n_2p1d_parameters1 = in_channels * 3 * 3 + 3 * out_channels
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1

        self.conv1_s = conv1x3x3(in_channels, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)

        self.conv1_t = conv3x1x1(mid_planes1, out_channels, stride)
        self.bn1_t = nn.BatchNorm3d(out_channels)

        n_3d_parameters2 = out_channels * out_channels * 3 * 3 * 3
        n_2p1d_parameters2 = out_channels * 3 * 3 + 3 * out_channels
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2

        self.conv2_s = conv1x3x3(out_channels, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)

        self.conv2_t = conv3x1x1(mid_planes2, out_channels)
        self.bn2_t = nn.BatchNorm3d(out_channels)

        self.downsample = nn.Sequential(
            conv1x1x1(in_channels, out_channels, stride),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.activation(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.activation(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.activation(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        residual = self.downsample(x)
        out += residual
        out = self.activation(out)

        return out


class PixelShuffle2p1D(nn.Module):
    def __init__(self, scale_factor: int):
        super(PixelShuffle2p1D, self).__init__()

        self.scale = scale_factor
        self.pxl = nn.PixelShuffle(self.scale)

    def forward(self, x: Tensor) -> Tensor:
        # b,c,t,h,w -> b,t,c,h,w
        x = x.permute(0, 2, 1, 3, 4)
        x = self.pxl(x)
        x = x.permute(0, 2, 1, 3, 4)
        return x


class Decoder2p1Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_t: int = None,
                 upsample_shape: tuple = None, upsample_w_h: int = 2,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(Decoder2p1Block, self).__init__()

        assert upsample_t or upsample_shape, "Either upsample_t or upsample_shape must be specified"

        if upsample_t:
            assert upsample_shape is None, "Cannot specify both upsample_t and upsample_shape"
            self.ups1_t = nn.Upsample(scale_factor=(upsample_t, 1, 1),
                                      mode='trilinear', align_corners=False)
            self.ups_res = nn.Upsample(scale_factor=(upsample_t, upsample_w_h, upsample_w_h),
                                       mode='trilinear', align_corners=False)
        elif upsample_shape:
            self.ups1_t = nn.Upsample(size=upsample_shape, mode='trilinear', align_corners=False)
            self.ups_res = nn.Upsample(size=upsample_shape, mode='trilinear', align_corners=False)

        self.activation = activation

        out1 = out_channels * (upsample_w_h**2)

        self.conv1_s = conv1x3x3(in_channels, out1, stride=1)
        self.pxl1_s = PixelShuffle2p1D(upsample_w_h)
        self.bn1_s = nn.BatchNorm3d(out_channels)

        self.conv1_t = conv3x1x1(out_channels, out_channels, stride=1)

        self.bn1_t = nn.BatchNorm3d(out_channels)

        self.conv2_s = conv1x3x3(out_channels, out_channels)
        self.bn2_s = nn.BatchNorm3d(out_channels)

        self.conv2_t = conv3x1x1(out_channels, out_channels)
        self.bn2_t = nn.BatchNorm3d(out_channels)

        self.conv_res = conv1x1x1(in_channels, out_channels, stride=1)

        self.bn_res = nn.BatchNorm3d(out_channels)

    def forward(self, x: Tensor) -> Tensor:

        residual = x

        out = self.conv1_s(x)
        out = self.pxl1_s(out)
        out = self.bn1_s(out)
        out = self.activation(out)

        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.activation(out)

        out = self.ups1_t(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.activation(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        residual = self.conv_res(residual)
        residual = self.ups_res(residual)
        residual = self.bn_res(residual)

        out += residual
        out = self.activation(out)

        return out


if __name__ == '__main__':
    import torch
    # x = torch.randn(16, 32, 4, 4, 4)
    x = torch.randn(32, 1, 20, 64, 64)
    enc = Encoder2p1Block(in_channels=1, out_channels=16, stride=1,)
    # dec = Decoder2p1Block(in_channels=32, out_channels=32, upsample_t=4)

    y = enc(x)

    print(y.shape)
