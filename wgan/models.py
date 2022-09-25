import torch
from torch import nn
from torch.nn import ConvTranspose2d, BatchNorm2d, ReLU, Tanh, Sigmoid, Conv2d, LeakyReLU

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, https://arxiv.org/pdf/1511.06434.pdf
        self.f = nn.Sequential(
            Conv2d(in_channels=1, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=512),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=1024),
            LeakyReLU(0.2, inplace=True),
            Conv2d(in_channels=1024, out_channels=1, kernel_size=4, bias=False),
            # Sigmoid()
        )


    def forward(self, x):
        return self.f(x)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, https://arxiv.org/pdf/1511.06434.pdf
        self.g = nn.Sequential(
            ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, bias=False),
            BatchNorm2d(num_features=1024),
            ReLU(True),
            ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=512),
            ReLU(True),
            ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(num_features=256),
            ReLU(True),
            ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            Tanh()
        )


    def forward(self, x):
        x = torch.unsqueeze(torch.unsqueeze(x, dim=-1), dim=-1)
        return self.g(x)
