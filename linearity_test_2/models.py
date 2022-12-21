import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d
from torchvision.models import resnet18

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 2
        self.d = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['data_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=scale*config['data_dim'], out_features=scale*config['data_dim'], bias=False),
            nn.BatchNorm1d(num_features=scale*config['data_dim']),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=scale*config['data_dim'], out_features=1)

            # 2 -> 3, scale=2
            # nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=scale*config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=1)

            # 8 -> 16, scale=4
            # nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=scale*config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=1)
        )

    def forward(self, x):
        return self.d(x)

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.e = nn.Sequential(

            # 2 -> 3
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']),

            # nn.BatchNorm1d(num_features=config['data_dim']),
            nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim'], bias=False),

            # 8 -> 16
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']),
        )


    def forward(self, x):
        return self.e(x)