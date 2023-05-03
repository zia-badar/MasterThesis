import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d, ReLU, BatchNorm1d, LeakyReLU

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 4
        self.d = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['encoding_dim']),
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

        self.e = nn.Sequential(
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=config['projection_dim'], out_features=config['encoding_dim']),
        )

    def forward(self, x):
        return self.e(x)

class Projection(nn.Module):

    def __init__(self, config):
        super(Projection, self).__init__()

        self.projection = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            nn.ReLU(),
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            nn.ReLU()
        )

        with open('results/projection_1', 'rb') as file:
            self.projection.load_state_dict(torch.load(file))

        # torch.save(self.projection.state_dict(), 'results/projection_1')

        # torch.save(self.projection.state_dict(), config['result_folder'] + 'projection')


    def forward(self, x):
        with torch.no_grad():
            return self.projection(x)