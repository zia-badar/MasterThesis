from torch import nn
from torch.nn.functional import leaky_relu


class Model(nn.Module):
    def __init__(self, model_type):
        super(Model, self).__init__()
        self.model_type = model_type

        if self.model_type == 'cifar':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=128, eps=1e-4, affine=False)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=256, eps=1e-4, affine=False)
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=256, eps=1e-4, affine=False)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1, bias=False)
            self.out = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = leaky_relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = leaky_relu(self.bn2(x))
        x = self.conv3(x)
        x = leaky_relu(self.bn3(x))
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.out(x)

        return x
