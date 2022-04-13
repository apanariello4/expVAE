import os
import sys
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops.layers.torch import Rearrange
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from model.block2p1d import Encoder2p1Block
from model.block2d import Encoder2DBlock
from model.learner import Learner


def get_inplanes():
    return [64, 128, 256, 512]


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        self.name = f'R2P1D-{sum(layers)*2+2}'

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(n_input_channels,
                                 mid_planes,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2),
                                 padding=(0, 3, 3),
                                 bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes,
                                 self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1),
                                 stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0),
                                 bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    @staticmethod
    def get_loss_function(**kwargs):
        return nn.CrossEntropyLoss()

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


def get_model_list() -> List[str]:
    return ["R2P1_18_K700", "R2P1_34_K700", "R2P1_50_K700",
            "R2P1_50_K700_M", "R2P1_18_K400", "R2P1_34_IG65",
            "R2P1_34_IG65_K"]


def get_model(model_conf: str, learner_layers: int = 0,
              dropout: Optional[float] = None, pretrained: bool = True,
              checkpoint_path: str = "/homes/apanariello/insectt/checkpoints"):
    """loads the model from the config string in format
        R2P1_layers_pretrain,
        "R2P1_18_K700", "R2P1_34_K700", from https://github.com/kenshohara/3D-ResNets-PyTorch,
        "R2P1_50_K700", "R2P1_50_K700_M" from https://github.com/kenshohara/3D-ResNets-PyTorch,
        "R2P1_18_K400" from torchvision
        "R2P1_34_IG65", "R2P1_34_IG65_K" from https://github.com/moabitcoin/ig65m-pytorch

    Args:
        model_conf (str): model configuration string
        learner_layers (int): if 0 uses 1 fc layer with batch norm, otherwise uses
            learner with <learner_layers> layers [2, 3]
        dropout (float): dropout rate when using learner
        checkpoint_path (str, optional): path to the checkpoint file.

    Returns:
        model (nn.Module): the model
        mean (tuple): mean of the dataset
        std (tuple): std of the dataset
    """
    assert model_conf in get_model_list()
    if learner_layers:
        assert dropout is not None
        assert learner_layers in (2, 3)
    if os.environ.get('USERNAME') == 'nello':
        checkpoint_path = "/home/nello/nas/checkpoints"

    if model_conf == "R2P1_18_K700":
        model = generate_model(18, n_classes=700)
        if pretrained:
            checkpoint = torch.load(
                f"{checkpoint_path}/r2p1d18_K_200ep.pth", map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
        mean = (0.4345, 0.4051, 0.3775)
        std = (0.2768, 0.2713, 0.2737)
        num_features_out = 512

    elif model_conf == "R2P1_34_K700":
        model = generate_model(34, n_classes=700)
        if pretrained:
            checkpoint = torch.load(
                f"{checkpoint_path}/r2p1d34_K_200ep.pth", map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
        mean = (0.4345, 0.4051, 0.3775)
        std = (0.2768, 0.2713, 0.2737)
        num_features_out = 512

    elif model_conf == "R2P1_50_K700":
        model = generate_model(50, n_classes=700)
        if pretrained:
            checkpoint = torch.load(
                f"{checkpoint_path}/r2p1d50_K_200ep.pth", map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
        mean = (0.4345, 0.4051, 0.3775)
        std = (0.2768, 0.2713, 0.2737)
        num_features_out = 2048

    elif model_conf == "R2P1_50_K700_M":
        model = generate_model(50, n_classes=1039)
        if pretrained:
            checkpoint = torch.load(
                f"{checkpoint_path}/r2p1d50_KM_200ep.pth", map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
        mean = (0.4345, 0.4051, 0.3775)
        std = (0.2768, 0.2713, 0.2737)
        num_features_out = 2048

    elif model_conf == "R2P1_18_K400":
        model = torchvision.models.video.__dict__["r2plus1d_18"](
            pretrained=pretrained
        )
        mean = (0.43216, 0.394666, 0.37645)
        std = (0.22803, 0.22145, 0.216989)
        num_features_out = 512

    elif model_conf == "R2P1_34_IG65":
        model = torch.hub.load("moabitcoin/ig65m-pytorch",
                               "r2plus1d_34_32_ig65m", num_classes=359, pretrained=pretrained)
        mean = (0.43216, 0.394666, 0.37645)
        std = (0.22803, 0.22145, 0.216989)
        num_features_out = 512

    elif model_conf == "R2P1_34_IG65_K":
        model = torch.hub.load("moabitcoin/ig65m-pytorch",
                               "r2plus1d_34_32_kinetics", num_classes=400, pretrained=pretrained)
        mean = (0.43216, 0.394666, 0.37645)
        std = (0.22803, 0.22145, 0.216989)
        num_features_out = 512

    if not learner_layers:
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=num_features_out),
            nn.Linear(num_features_out, 1),
            nn.Sigmoid(),
        )
    else:
        model.fc = Learner(layers=learner_layers,
                           input_dim=num_features_out,
                           drop_p=dropout)

    model.name = model_conf
    model.count_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, mean, std


class SmallResNet(nn.Module):
    def __init__(self):
        super(SmallResNet, self).__init__()
        # input 1x20x64x64
        self.name = "SmallResNet"
        self.net = nn.Sequential(
            Rearrange("b c t h w -> (b t) c h w"),
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Rearrange("(b t) c h w -> b c t h w", t=20),
            nn.Conv3d(16, 32, kernel_size=5, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=5, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d(output_size=(1, 8, 8)),
            Rearrange("b c t h w -> b (c t h w)"),
            nn.Linear(64 * 1 * 8 * 8, 2),
        )

    def forward(self, x):
        x = self.net(x)
        return x

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torchsummary
    model = SmallResNet()
    #print(torchsummary.summary(model, (1, 20, 64, 64), 16))
    print(f'{model.num_parameters:,}')
    x = torch.randn(16, 1, 20, 64, 64)
    y = model(x)
    print(y.shape)
