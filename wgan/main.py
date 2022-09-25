from cmath import sqrt

import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms, Resize
from tqdm import tqdm

from wgan.models import Discriminator, Generator


def train(config):
    dataset = MNIST(root='../', train=True, transform=transforms.Compose([ToTensor(), Resize(size=(32, 32)), transforms.Normalize((0.5, ), (0.5, ))]), download=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20, drop_last=True)

    f = Discriminator().cuda()
    g = Generator().cuda()
    f.train()
    g.train()

    optim_f = RMSprop(f.parameters(), lr=config['learning_rate'])
    optim_g = RMSprop(g.parameters(), lr=config['learning_rate'])
    z_dist = MultivariateNormal(torch.zeros(100), torch.eye(100))
    for epoch in tqdm(range(config['epochs'])):
        for i, (x, l) in enumerate(dataloader):
            x = x.cuda()
            z = z_dist.sample_n(config['batch_size']).cuda()
            optim_f.zero_grad()
            loss = -(torch.mean(f(x)) - torch.mean(f(g(z))))
            loss.backward()
            optim_f.step()

            for p in f.parameters():
                p.data = p.data.clamp(-config['clip'], config['clip'])

            if i % config['n_critic'] == 0:
                z = z_dist.sample_n(config['batch_size']).cuda()
                optim_g.zero_grad()
                loss = -torch.mean(f(g(z)))
                loss.backward()
                optim_g.step()

        evaluate(g)

def evaluate(model):
    z_dist = MultivariateNormal(torch.zeros(100), torch.eye(100))
    with torch.no_grad():
        model.eval()
        z = z_dist.sample_n(config['batch_size']).cuda()
        samples = model(z)
        stack = (int)(sqrt(config['batch_size']).real)
        samples = samples.reshape(stack, stack, 32, 32).permute(0, 2, 1, 3).reshape(stack*32, stack*32).cpu().numpy()

        _, ax = plt.subplots()
        ax.imshow(samples)
        plt.show()

    model.train()

if __name__ == '__main__':

    config = {'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e6)}

    train(config)