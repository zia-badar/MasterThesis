import torch
from torch import nn
from torch.nn.functional import leaky_relu


class Model(nn.Module):

    def __init__(self, model_type=None):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=8, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5), padding=(2, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=4, eps=1e-4, affine=False)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(in_features=4*7*7, out_features=32, bias=False)

    def forward(self, batch):
        x = self.conv1(batch)
        x = self.pool(leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(leaky_relu(self.bn2(x)))
        x = self.flatten(x)
        x = self.out(x)
        return x

    def get_weights(self):
        return torch.zeros(1)