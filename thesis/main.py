from cmath import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, transforms, Resize
from tqdm import tqdm

from thesis.dataset import OneClassDatasetWrapper
from thesis.models import Encoder, Discriminator

def train(config):
    dataset = OneClassDatasetWrapper(MNIST(root='../', train=True, download=True), 1, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width']))]))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)

    f = Discriminator(config).cuda()
    e = Encoder(config).cuda()
    f.train()
    e.train()

    optim_f = RMSprop(f.parameters(), lr=config['learning_rate'])
    optim_e = RMSprop(e.parameters(), lr=config['learning_rate'])
    z_dist = MultivariateNormal(torch.zeros(config['z_dim']), torch.eye(config['z_dim']))

    def pretty_tensor(tensor):
        return np.array_repr(tensor.detach().cpu().numpy()).replace('\n', '')
    resize = Resize(size=(config['height'], config['width']))

    train_progress_bar = tqdm(range(config['epochs']))
    iter = 0
    for _ in train_progress_bar:
        for (x, l) in dataloader:
            x = x.cuda()
            if iter % (config['n_critic'] + 1) != 0:
                z = z_dist.sample_n(config['batch_size']).cuda()

                optim_f.zero_grad()
                loss = -(torch.mean(f(z)) - torch.mean(f(e(x))))
                loss.backward()
                optim_f.step()

                for p in f.parameters():
                    p.data = p.data.clamp(-config['clip'], config['clip'])


                # train_progress_bar.desc = f'em distance: {loss.item() : .8f}'

            if iter % (config['n_critic'] + 1) == 0:
                optim_e.zero_grad()
                e_x = e(x)
                loss = -torch.mean(f(e_x))
                # loss = -torch.mean(f(e_x))
                loss.backward()
                optim_e.step()

                train_progress_bar.set_description(f'encoding_mean: {pretty_tensor(torch.round(torch.mean(e_x, dim=0), decimals=4))}, encoding_variance: {pretty_tensor(torch.round(torch.var(e_x, dim=0, unbiased=False), decimals=4))}, encoding co-variance: {pretty_tensor(torch.round(torch.cov(e_x.T, correction=0), decimals=4))}')


                if iter % ((config['n_critic'] +1)*5) == 0 and False:
                    original_batch_size = config['batch_size']
                    config['batch_size'] = 1
                    x = x[:1]
                    heat_map = e.heat_map(x).detach()
                    heat_map = resize(heat_map)

                    stack = (int)(sqrt(config['batch_size']).real)
                    img = x.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape( stack * config['height'], stack * config['width']).cpu().numpy()

                    heat_map_img = heat_map.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape( stack * config['height'], stack * config['width']).cpu().numpy()

                    _, ax = plt.subplots(2)
                    ax[0].imshow(img)
                    ax[1].imshow(heat_map_img)
                    plt.show()
                    config['batch_size'] = original_batch_size

            torch.cuda.empty_cache()
            iter += 1

if __name__ == '__main__':
    config = {'height': 32, 'width': 32, 'batch_size': 128, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e7), 'z_dim': 3}

    train(config)