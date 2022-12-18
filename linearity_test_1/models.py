import torch
from torch import nn
from torch.nn import Flatten, Conv2d, BatchNorm2d
from torchvision.models import resnet18


class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()

        model = resnet18()
        layers = []
        for layer_name, layer in model.named_children():
            if layer_name == 'avgpool':
                layers.append(Conv2d(512, 128, kernel_size=(1, 1), bias=False))
                layers.append(BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(Flatten())
                layers.append(nn.Linear(in_features=128, out_features=num_classes))
                break
            else:
                layers.append(layer)

        self.classification_space = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(*layers[-1:])

    def forward(self, x):
        classification_space = self.classification_space(x)
        out = self.classifier(classification_space)
        return classification_space, out

class Discriminator(nn.Module):
    def __init__(self, encoding_dim):
        super(Discriminator, self).__init__()

        self.d = nn.Sequential(
            nn.Linear(in_features=encoding_dim, out_features=encoding_dim*(2**1), bias=False),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
            nn.Linear(in_features=encoding_dim*(2**1), out_features=encoding_dim*(2**2), bias=False),
            nn.BatchNorm1d(num_features=encoding_dim*(2**2)),
            # nn.Tanh(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoding_dim*(2**2), out_features=encoding_dim*(2**3), bias=False),
            nn.BatchNorm1d(num_features=encoding_dim*(2**3)),
            # nn.Tanh(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=encoding_dim*(2**3), out_features=1, bias=False)
        )

    def forward(self, x):
        return self.d(x)

class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()

        self.e = nn.Sequential(
            # nn.Linear(in_features=128, out_features=128, bias=False),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(num_features=128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.BatchNorm1d(num_features=128),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=8, bias=False),
            # nn.BatchNorm1d(num_features=16),


            # nn.Tanh(),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=128, out_features=128),
            # nn.BatchNorm1d(num_features=128),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=128, out_features=128),
            # nn.BatchNorm1d(num_features=128),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(in_features=128, out_features=encoding_dim)
            nn.Tanh()
        )

        # list(self.e.parameters())[0].data = torch.eye(128)

    def forward(self, x):
        return self.e(x)