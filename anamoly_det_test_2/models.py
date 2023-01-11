from pickle import load

import torch
from torch import nn
from torch.nn import ReLU, BatchNorm1d
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        # self.g = nn.Sequential(
        #                        nn.Linear(512, 512, bias=False),
        #                        nn.BatchNorm1d(512),
        #                        nn.ReLU(inplace=True),
        #                        nn.Linear(512, feature_dim),
        #                        )

        layers = []
        for _ in range(8):
            layers += [nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(512, 128))
        self.g = nn.Sequential(*layers)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out


class AbsActivation(nn.Module):

    def __init__(self, base=0.001, slope=0.001):
        super(AbsActivation, self).__init__()

        self.base = base
        self.slope = slope

    def forward(self, x):
        ret = self.base + torch.abs(x) * self.slope
        return ret

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 1
        self.d = nn.Sequential(

            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            BatchNorm1d(num_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=config['encoding_dim']),
            BatchNorm1d(num_features=config['encoding_dim']),
            ReLU(inplace=True),
            nn.Linear(in_features=config['encoding_dim'], out_features=1)

        )

    def forward(self, x):
        return torch.sum(self.d(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        layers = []
        for _ in range(3):
            layers += [
                nn.Linear(in_features=config['data_dim'], out_features=config['data_dim']),
                nn.BatchNorm1d(num_features=config['data_dim']),
        #         AbsActivation(slope=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        layers.append(nn.Linear(in_features=config['data_dim'], out_features=config['encoding_dim']))

        self.e = nn.Sequential(*layers)

    def forward(self, x):
        return self.e(x)

class Projection(nn.Module):

    def __init__(self, config):
        super(Projection, self).__init__()

        # self.projection_1 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.projection_2 = torch.rand(size=(config['encoding_dim'], config['data_dim']))
        # self.translation_1 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))
        # self.translation_2 = 5 * torch.normal(mean=0, std=1, size=(config['data_dim'],))
        
        self.projection_1 = torch.tensor([[0.8560, 0.3906, 0.7770], [0.1772, 0.2052, 0.4125], [0.7921, 0.8567, 0.3301]])
        self.projection_2 = torch.tensor([[0.9458, 0.2717, 0.7411], [0.5602, 0.7715, 0.1062], [0.4664, 0.5055, 0.6179]])
        self.translation_1 = torch.tensor([-2.6553,  2.4304,  8.3437])
        self.translation_2 = torch.tensor([5.3460, 2.8367, 1.0990])
        #
        # with open('results/result_non_l', 'rb') as file:
        #     result = load(file)
        #
        # self.projection = result.projection.projection
        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=config['encoding_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(in_features=config['data_dim'], out_features=config['data_dim']),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )

    def forward(self, x):
        projections = torch.empty((x.shape[0], ) + self.projection_1.shape)
        translations = torch.empty((x.shape[0], ) + self.translation_1.shape)
        seed = (torch.rand((x.shape[0],)) < 0.5)
        translations[seed] = self.translation_1
        translations[torch.logical_not(seed)] = self.translation_2
        projections[seed] = self.projection_1
        projections[torch.logical_not(seed)] = self.projection_2
        return torch.squeeze(x[:, None, :] @ projections) + translations
        #
        # with torch.no_grad():
        #     return self.projection(x)