from cmath import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize
from tqdm import tqdm

from wgan.dataset import RandomBoxDataset, OneClassDataset
from wgan.dcgan import DCGAN_D, DCGAN_G
from wgan.models import Discriminator, Generator


def train(config):
    if config['dataset'] == 'mnist':
        dataset = MNIST(root='../', train=True, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True)
    if config['dataset'] == 'cifar':
        dataset = OneClassDataset(CIFAR10(root='../', train=True, download=True), one_class_labels=[4], transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif config['dataset'] == 'random_box':
        dataset = RandomBoxDataset()
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    f = Discriminator().cuda()
    g = Generator(config).cuda()
    f.apply(weights_init)
    g.apply(weights_init)

    # f = DCGAN_D(isize=64, nz=100, nc=3, ndf=64, ngpu=1).cuda()
    # g = DCGAN_G(isize=64, nz=100, nc=3, ngf=64, ngpu=1).cuda()
    f.train()
    g.train()

    optim_f = RMSprop(f.parameters(), lr=config['learning_rate'])
    optim_g = RMSprop(g.parameters(), lr=config['learning_rate'])
    z_dist = MultivariateNormal(torch.zeros(config['z_dim']), torch.eye(config['z_dim']))
    progress_bar = tqdm(range(config['epochs']))
    gen_iter = 0
    for _ in progress_bar:
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
                for p in f.parameters():
                    p.requires_grad = False

                z = z_dist.sample_n(config['batch_size']).cuda()
                optim_g.zero_grad()
                loss = -torch.mean(f(g(z)))
                loss.backward()
                optim_g.step()

                for p in f.parameters():
                    p.requires_grad = True

                gen_iter += 1
                progress_bar.set_description(f'gen_iter: {gen_iter}')
                if gen_iter % 1000 == 0:
                    evaluate(g, config)
                    torch.save(g.state_dict(), f'model_ {gen_iter}')

def evaluate(model, config):
    z_dist = MultivariateNormal(torch.zeros(config['z_dim']), torch.eye(config['z_dim']))
    with torch.no_grad():
        model.eval()
        z = z_dist.sample_n(config['batch_size']).cuda()
        samples = model(z)
        stack = (int)(sqrt(config['batch_size']).real)
        samples = samples.permute(1, 0, 2, 3).reshape(3, stack, stack, config['height'], config['width']).permute(0, 1, 3, 2, 4).reshape(3, stack*config['height'], stack*config['width']).cpu().numpy()

        samples = (samples + 1) / 2
        samples = np.clip(samples, 0., 1.)

        _, ax = plt.subplots()
        ax.imshow(np.transpose(samples, (1, 2, 0)))
        plt.show()

    model.train()

if __name__ == '__main__':

    # config = {'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e7), 'height': 32, 'width': 32, 'z_dim': 100, 'dataset': 'mnist'}
    # config = {'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e7), 'height': 32, 'width': 32, 'z_dim': 100, 'dataset': 'random_box'}
    config = {'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e7), 'height': 64, 'width': 64, 'z_dim': 100, 'dataset': 'cifar'}

    train(config)