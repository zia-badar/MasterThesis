import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d, ReLU, BatchNorm1d, LeakyReLU

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        scale = 2
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

        scale = 4
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

        self.projection = nn.Sequential(
            nn.Linear(in_features=config['encoding_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            nn.ReLU(),
            nn.Linear(in_features=config['projection_dim'], out_features=config['projection_dim']),
            BatchNorm1d(num_features=config['projection_dim']),
            nn.ReLU()
        )

        # with open('results/projection_2_disconnected', 'rb') as file:
        #     self.projection.load_state_dict(torch.load(file))

        # torch.save(self.projection.state_dict(), 'results/projection_1')

        # torch.save(self.projection.state_dict(), config['result_folder'] + 'projection')

    def disconnect_1d(self, x):
        # 1d discconect


        projection_1 = torch.tensor([[0.8560, 0.3906]])
        projection_2 = torch.tensor([[0.9458, 0.2717]])
        translation_1 = torch.tensor([-2.6553,  2.4304])
        translation_2 = torch.tensor([5.3460, 2.8367])

        # disconnect = torch.tensor([[1., -1.]])
        # project_2d = torch.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]])
        # translate = torch.tensor([[1, 0, -.5, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, -.5], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        # rotate_90 = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.]])
        # translate_2 = torch.tensor([[1, 0, 5., 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 5.], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
        # project_1d = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

        # mask = torch.cat([(x > 0).repeat(1, 3), (x < 0).repeat(1, 3)], dim=-1)
        # x = x @ disconnect
        # x = torch.relu(x)
        # x = x @ project_2d
        # #translate
        # x = torch.cat([x[:, :2], torch.ones_like(x[:, :1]), x[:, 2:], torch.ones_like(x[:, :1])], dim=-1)
        # x = (x @ translate.t()) * mask*1.
        # x = torch.cat([x[:, :2], x[:, 3:-1]], dim=-1)
        # #rotate
        # x = x @ rotate_90.t()
        # #translate
        # x = torch.cat([x[:, :2], torch.ones_like(x[:, :1]), x[:, 2:], torch.ones_like(x[:, :1])], dim=-1)
        # x = (x @ translate_2.t()) * mask*1.
        # x = torch.cat([x[:, :2], x[:, 3:-1]], dim=-1)

        # x = x @ project_1d

        projections = torch.empty((x.shape[0],) + projection_1.shape)
        translations = torch.empty((x.shape[0],) + translation_1.shape)
        seed = (x[:, 0] < 0)
        translations[seed] = translation_1
        translations[torch.logical_not(seed)] = translation_2
        projections[seed] = projection_1
        projections[torch.logical_not(seed)] = projection_2
        return torch.squeeze(x[:, None, :] @ projections) + translations

        return x

    def forward(self, x):
        with torch.no_grad():

            return self.disconnect_1d(x)