from torch import nn
from torchvision.models import resnet18


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        nonlocal config

        # modified resnet18 for cifar10, https://github.com/leftthomas/SimCLR/blob/master/model.py
        f = []
        for name, layer in resnet18().named_children():
            if name == 'conv1':
                layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(layer, nn.Linear) or not isinstance(layer, nn.MaxPool2d):
                f.append(layer)

        self.f = nn.Sequential(*f)

        self.p_s_x = nn.Sequential(nn.Linear(2048, config.shift_count), nn.Softmax(dim=1))

        self.z = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, config.encoding_dim))

    def forward(self, x):
        f = self.f(x)
        p_s_x = self.p_s_x(f)
        z = self.z(f)

        return z, p_s_x
