import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d
from torchvision.models import resnet18

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        encoding_dim = config['encoding_dim']
        self.d = nn.Sequential(
            nn.Linear(in_features=encoding_dim, out_features=encoding_dim*(2**1)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=encoding_dim*(2**1), out_features=encoding_dim*(2**2), bias=False),
            nn.BatchNorm1d(num_features=encoding_dim*(2**2)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=encoding_dim*(2**2), out_features=encoding_dim*(2**3), bias=False),
            nn.BatchNorm1d(num_features=encoding_dim*(2**3)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=encoding_dim*(2**3), out_features=1)
        )

    def forward(self, x):
        return self.d(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.e = nn.Sequential(
            nn.BatchNorm1d(num_features=config['data_dim']),
            nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            nn.BatchNorm1d(num_features=config['data_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            nn.BatchNorm1d(num_features=config['data_dim']),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']),
        )


    def forward(self, x):
        return self.e(x)