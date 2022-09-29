import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import transforms, ToTensor, Resize
from tqdm import tqdm

from GradCam.model import Model


def train():
    dataset = MNIST(root='../', train=True, transform=transforms.Compose([ToTensor(), Resize(size=(32, 32))]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = Model(num_classes=10).cuda()
    model.train()
    loss = CrossEntropyLoss()           # BCE/BCEwithLogit is used in 2-class multiclass or n-class multilabel classification problem and CrossEntopyLoss is used in n-class multiclass
    optim = Adam(model.parameters(), lr=1e-3)

    for _ in tqdm(range(1), desc='epochs'):
        for (x, l) in dataloader:
            x = x.cuda()
            target = l.cuda()
            predicted = model(x)

            model.zero_grad()
            l = loss(predicted, target)
            l.backward()
            optim.step()

    return model

def evaluate(model):
    dataset = MNIST(root='../', train=False, transform=transforms.Compose([ToTensor(), Resize(size=(32, 32))]))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=20)
    model.eval()

    tp = 0
    all = 0
    with torch.no_grad():
        for (x, l) in dataloader:
            x = x.cuda()
            target = l.cuda()
            predicted = model(x)

            predicted = torch.argmax(predicted, dim=1)

            tp += torch.sum(target == predicted).item()
            all += target.shape[0]

    print(f'accuracy: {tp/all : .2f}')
    model.train()


if __name__ == '__main__':

    model = train()
    evaluate(model)
