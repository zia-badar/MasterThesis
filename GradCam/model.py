import torch
from torch import nn, Tensor
from torch.nn import ReLU
from torch.nn.functional import one_hot
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import BasicBlock

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


    def grad_cam(self, x, l):
        l = one_hot(l, num_classes=self.num_classes)
        logit = self(x)
        self.zero_grad()
        loss = torch.sum(logit * l)
        self.activation_grad = None
        loss.backward()

        alpha = torch.mean(self.activation_grad, dim=(2, 3))
        grad_cam = self.relu(torch.sum(alpha[:, :, None, None] * self.activation, dim=1))

        return grad_cam

class CustomResNet(GradCamNet, ResNet):     # resnet18 with only first 4 layers
    def __init__(self, num_classes):
        super().__init__(num_classes, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Linear(64, num_classes)

        self.layer1.register_forward_hook(self._forward_hook)
        self.layer1.register_backward_hook(self._backward_hook)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class SimpleConvNet(GradCamNet, nn.Module):

    def __init__(self, num_classes):
            super().__init__(num_classes)
            self.relu = ReLU(inplace=True)

            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
            self.relu1 = ReLU(inplace=True)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.relu2 = ReLU()
            self.relu2.register_forward_hook(self._forward_hook)
            self.relu2.register_backward_hook(self._backward_hook)
            self.max_pool = nn.MaxPool2d(kernel_size=2)
            self.drop_out1 = nn.Dropout(0.5)
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(9216, 128)
            self.relu3 = ReLU(inplace=True)
            self.drop_out2 = nn.Dropout(0.1)
            self.dense2 = nn.Linear(128, 10)


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool(x)
        x = self.drop_out1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.drop_out2(x)
        x = self.dense2(x)
        return x