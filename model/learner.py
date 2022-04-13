import torch.nn as nn


class Learner(nn.Module):
    def __init__(self, layers: int, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        if layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 1),
                nn.Sigmoid(),
            )
        elif layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                nn.ReLU(),
                nn.Dropout(drop_p),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.classifier(x)
