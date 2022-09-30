from cmath import sqrt

import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms, ToTensor, Resize
from tqdm import tqdm
from model import CustomResNet, SimpleConvNet


def train(config):
    dataset = MNIST(root='../', train=True, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width']))]))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model = config['model'](num_classes=10).cuda()
    model.train()
    loss = CrossEntropyLoss()           # BCE/BCEwithLogit is used in 2-class multiclass or n-class multilabel classification problem and CrossEntopyLoss is used in n-class multiclass
    optim = Adam(model.parameters(), lr=1e-3)

    for _ in tqdm(range(1), desc='epochs'):
        for (x, l) in dataloader:
            x = x.cuda()
            l = l.cuda()
            target = l
            predicted = model(x)

            model.zero_grad()
            los = loss(predicted, target)
            los.backward()
            optim.step()

        evaluate(model, config, sample_evaluate=True)

    return model

def evaluate(model, config, sample_evaluate=False):
    batch_size = config['batch_size']
    config['batch_size'] = 4
    dataset = MNIST(root='../', train=False, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width']))]))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], drop_last=True)
    model.eval()

    tp = 0
    all = 0
    resize = Resize(size=(config['height'], config['width']))
    for (x, l) in dataloader:
        x = x.cuda()
        l = l.cuda()
        target = l
        predicted = model(x)

        predicted = torch.argmax(predicted, dim=1)

        tp += torch.sum(target == predicted).item()
        all += target.shape[0]

        if sample_evaluate:
            heat_map = model.grad_cam(x, l).detach()
            heat_map = resize(heat_map)

            stack = (int)(sqrt(config['batch_size']).real)
            img = x.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape(stack * config['height'], stack * config['width']).cpu().numpy()

            heat_map_img = heat_map.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape(stack * config['height'], stack * config['width']).cpu().numpy()

            _, ax = plt.subplots(2)
            ax[0].imshow(img)
            ax[1].imshow(heat_map_img)
            plt.show()
            break

    if not sample_evaluate:
        print(f'accuracy: {tp/all : .2f}')
    model.train()
    config['batch_size'] = batch_size


if __name__ == '__main__':

    # config = {'height': 224, 'width': 224, 'batch_size': 64, 'model': CustomResNet}
    config = {'height': 28, 'width': 28, 'batch_size': 64, 'model': SimpleConvNet}

    model = train(config)
    torch.save(model.state_dict(), 'saved_model')

    # model = CustomResNet(num_classes=10).cuda()
    # model.load_state_dict(torch.load('saved_model'))

    config['batch_size'] = 4
    evaluate(model, config)
