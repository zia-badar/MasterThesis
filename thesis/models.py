import torch
import torchvision.models
from torch import nn, Tensor
from torch.distributions.constraints import one_hot
from torch.nn import ConvTranspose2d, BatchNorm2d, ReLU, Tanh, Sigmoid, Conv2d, LeakyReLU, BatchNorm1d, Flatten, Linear
from resnet_modified import ResNet, BasicBlock


class Discriminator(nn.Module):

    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.f = nn.Sequential(
            nn.Linear(in_features=config['z_dim'], out_features=64),
            # BatchNorm1d(num_features=16),
            # Tanh(),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=64, out_features=128),
            BatchNorm1d(num_features=128),
            # Tanh(),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=128, out_features=256),
            BatchNorm1d(num_features=256),
            # Tanh(),
            LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )


    def forward(self, x):
        return torch.sigmoid(self.f(x))

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        
        # self.e = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=config['z_dim'])

        self.e = CustomResNet(num_classes=config['z_dim'])

        # # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, https://arxiv.org/pdf/1511.06434.pdf
        # self.e = nn.Sequential(
        #     Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
        #     BatchNorm2d(num_features=128),
        #     LeakyReLU(0.2, inplace=True),
        #     Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
        #     BatchNorm2d(num_features=256),
        #     LeakyReLU(0.2, inplace=True),
        #     Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
        #     BatchNorm2d(num_features=512),
        #     LeakyReLU(0.2, inplace=True),
        #     Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1, bias=False),
        #     BatchNorm2d(num_features=1024),
        #     LeakyReLU(0.2, inplace=True),
        #     Conv2d(in_channels=1024, out_channels=config['z_dim'], kernel_size=4, bias=False),
        #     Tanh(),
        #     Flatten()
        # )


    def forward(self, x):
        x = self.e(x)
        return x

    def heat_map(self, x):
        return self.e.grad_cam(x)
        # return x


class GradCamNet:
    def __init__(self, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = n_classes

        # https://stackoverflow.com/questions/65011884/understanding-backward-hooks
        # https://github.com/pytorch/pytorch/issues/598#issuecomment-275489803
        def _backward_hook(module, input, output, _self=self):
            _self.activation_grad = output[0]

        def _forward_hook(module, input, output, _self=self):
            _self.activation = output

        self._backward_hook = _backward_hook
        self._forward_hook = _forward_hook


    def grad_cam(self, x):
        logit = self(x)
        self.zero_grad()
        loss = torch.sum(abs(logit))
        loss.backward()

        alpha = torch.mean(self.activation_grad, dim=(2, 3))
        grad_cam = self.relu(torch.sum(alpha[:, :, None, None] * self.activation, dim=1))

        return grad_cam

class CustomResNet(GradCamNet, ResNet):     # resnet18 with only first 4 layers
    def __init__(self, num_classes):
        super().__init__(num_classes, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(128, num_classes)

        self.layer2.register_forward_hook(self._forward_hook)
        self.layer2.register_backward_hook(self._backward_hook)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

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