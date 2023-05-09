import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d, ReLU, BatchNorm1d, LeakyReLU

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 6
        self.d = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=scale*config['encoding_dim'], out_features=scale*config['encoding_dim']),
            BatchNorm1d(num_features=scale*config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=scale*config['encoding_dim'], out_features=scale*config['encoding_dim']),
            BatchNorm1d(num_features=scale*config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=scale*config['encoding_dim'], out_features=1)
        )

    def forward(self, x):
        return self.d(x)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        scale = 6
        self.e = nn.Sequential(
            nn.Linear(in_features=config['projection_dim'], out_features=scale*config['projection_dim']),
            BatchNorm1d(num_features=scale*config['projection_dim']),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=scale*config['projection_dim'], out_features=scale*config['projection_dim']),
            BatchNorm1d(num_features=scale*config['projection_dim']),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=scale*config['projection_dim'], out_features=scale*config['projection_dim']),
            BatchNorm1d(num_features=scale*config['projection_dim']),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=scale*config['projection_dim'], out_features=config['encoding_dim']),
        )

    def forward(self, x):
        return self.e(x)

class Projection(nn.Module):

    def __init__(self, config):
        super(Projection, self).__init__()
        self.config = config

        self.projection_1 = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=config['projection_dim']),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.projection_2 =  nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=config['projection_dim']),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        with open(f'results/projections/{config["manifold_type"]}/{config["encoding_dim"]}d_{config["projection_dim"]}d/projection', 'rb') as file:
            self.load_state_dict(torch.load(file))

        # torch.save(self.state_dict(), 'results/projection')

    def forward(self, x):
        with torch.no_grad():
            if self.config['manifold_type'] == 'connected':
                return self.projection_1(x)
            elif self.config['manifold_type'] == 'disconnected':
                return torch.where(torch.unsqueeze(x[:, 0] < 0, -1), self.projection_1(x), self.projection_2(x))