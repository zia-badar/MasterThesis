from pickle import load

import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d
from torchvision.models import resnet18

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 1
        self.d = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=scale * config['data_dim']),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(in_features=scale * config['data_dim'], out_features=scale * config['data_dim'], bias=False),
            nn.BatchNorm1d(num_features=scale * config['data_dim']),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Linear(in_features=scale * config['data_dim'], out_features=1, bias=False)

            # 3 -> 3
            # nn.Linear(in_features=config['encoding_dim'], out_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=scale*config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=scale*config['data_dim']),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_features=scale*config['data_dim'], out_features=1)

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
        self.b = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        return torch.abs(self.d(x)) + self.b


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        scale = 0
        self.e = nn.Sequential(

            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'] + scale, bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim'] + scale),
            # nn.Linear(in_features=config['data_dim'] + scale, out_features=config['data_dim'] + scale, bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim'] + scale),
            # nn.Linear(in_features=config['data_dim'] + scale, out_features=config['encoding_dim']),

            nn.Linear(in_features=config['data_dim'], out_features=config['data_dim']+scale),
            nn.BatchNorm1d(num_features=config['data_dim']+scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=config['data_dim']+scale, out_features=config['encoding_dim']),

            # 3 -> 3, scale = 1
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'] + scale, bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim'] + scale),
            # nn.Linear(in_features=config['data_dim'] + scale, out_features=config['encoding_dim']),

            # 2 -> 3
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']),

            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim'], bias=False),

            # 8 -> 16
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.Linear(in_features=config['data_dim'], out_features=config['data_dim'], bias=False),
            # nn.BatchNorm1d(num_features=config['data_dim']),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']),
        )


    def forward(self, x):
        return self.e(x)

class Projection(nn.Module):

    def __init__(self, config):
        super(Projection, self).__init__()

        # self.projection_1 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.projection_2 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.translation_1 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))
        # self.translation_2 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))
        
        # self.projection_1 = torch.tensor([[0.8560, 0.3906, 0.7770], [0.1772, 0.2052, 0.4125], [0.7921, 0.8567, 0.3301]])
        # self.projection_2 = torch.tensor([[0.9458, 0.2717, 0.7411], [0.5602, 0.7715, 0.1062], [0.4664, 0.5055, 0.6179]])
        # self.translation_1 = torch.tensor([-2.6553,  2.4304,  8.3437])
        # self.translation_2 = torch.tensor([5.3460, 2.8367, 1.0990])

        with open('results/result_non_l', 'rb') as file:
            result = load(file)

        self.projection = result.projection.projection
        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=config['encoding_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(in_features=config['data_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

    def forward(self, x):
        # projections = torch.empty((x.shape[0], ) + self.projection_1.shape)
        # translations = torch.empty((x.shape[0], ) + self.translation_1.shape)
        # seed = (torch.rand((x.shape[0],)) < 0.5)
        # translations[seed] = self.translation_1
        # translations[torch.logical_not(seed)] = self.translation_2
        # projections[seed] = self.projection_1
        # projections[torch.logical_not(seed)] = self.projection_2
        # return torch.squeeze(x[:, None, :] @ projections) + translations

        with torch.no_grad():
            return self.projection(x)