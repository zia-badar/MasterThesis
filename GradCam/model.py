from torch import nn
from torchvision.models import resnet18


class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.resnet = resnet18(num_classes=num_classes)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def __call__(self, x):
        return self.resnet(x)