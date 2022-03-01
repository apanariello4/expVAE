import torch.nn as nn


class ResidualLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super(ResidualLinear, self).__init__()

        mid_features = out_features // 2
        # TODO batchnorm?
        self.fc1 = nn.Linear(in_features, mid_features, bias=bias)
        self.activation = activation
        self.fc2 = nn.Linear(mid_features, out_features, bias=bias)

        self.residual = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        residual = self.residual(x)

        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)

        return self.activation(out + residual)
