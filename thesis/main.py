from cmath import sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.distributions import MultivariateNormal, Uniform, Normal
from torch.linalg import norm
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize
from tqdm import tqdm

from thesis.dataset import RandomBoxDataset, OneClassDataset
from thesis.models import Encoder, Discriminator

def train(config):
    inlier = [5]
    outlier = list(range(10))
    outlier.remove(5)
    dataset = OneClassDataset(MNIST(root='../', train=True, download=True), one_class_labels=inlier, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5), (0.5))]))
    split_size = (int)(.7*len(dataset))
    dataset_splits = random_split(dataset, [split_size, len(dataset) - split_size])
    dataset = dataset_splits[0]
    _dataset = OneClassDataset(MNIST(root='../', train=True, download=True), zero_class_labels=outlier, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5), (0.5))]))
    val_dataset = ConcatDataset([_dataset, dataset_splits[1]])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    f = Discriminator(config).cuda()
    e = Encoder(config).cuda()
    f.train()
    e.train()
    f.apply(weights_init)
    e.apply(weights_init)

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

                # a = torch.round(torch.cov(e_x.T, correction=0), decimals=4)
                # b = (torch.sum(torch.abs(a)) - torch.sum(torch.abs(torch.diag(a)))).item()
                # train_progress_bar.set_description(f'encoding_mean: {pretty_tensor(torch.round(torch.mean(e_x, dim=0), decimals=4))}, encoding_variance: {pretty_tensor(torch.round(torch.var(e_x, dim=0, unbiased=False), decimals=4))}, encoding co-variance: {pretty_tensor(a)}, {b}')

                if iter % ((config['n_critic'] + 1)*50*20) == 0:
                    with torch.no_grad():
                        e.eval()
                        targets = []
                        scores = []
                        zs = []
                        train_dl = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
                        for _x, _l in train_dl:
                            _x = _x.cuda()
                            zs.append(e(_x))

                        zs = torch.cat(zs)
                        mean = torch.mean(zs, dim=0)
                        var = torch.cov(zs.T, correction=0)
                        dist = MultivariateNormal(loc = mean, covariance_matrix=var)

                        for _x, _l in val_dataloader:
                            _x = _x.cuda()
                            z = e(_x)
                            scores.append(dist.log_prob(z))
                            targets.append(_l)


                        # for _x, _l in val_dataloader:
                        #     _x = _x.cuda()
                        #     z = e(_x)
                        #     scores.append(norm(z, dim=1))
                        #     targets.append(_l)

                        scores = torch.cat(scores).cpu().numpy()
                        targets = torch.cat(targets).numpy()

                        print(f'iter: {iter}, roc: {roc_auc_score(targets, scores)}, mean: {mean}, cov: {var}')

                        e.train()



                # if iter % ((config['n_critic'] +1)*5) == 0 and False:
                #     original_batch_size = config['batch_size']
                #     config['batch_size'] = 1
                #     x = x[:1]
                #     heat_map = e.heat_map(x).detach()
                #     heat_map = resize(heat_map)
                #
                #     stack = (int)(sqrt(config['batch_size']).real)
                #     img = x.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape( stack * config['height'], stack * config['width']).cpu().numpy()
                #
                #     heat_map_img = heat_map.reshape(stack, stack, config['height'], config['width']).permute(0, 2, 1, 3).reshape( stack * config['height'], stack * config['width']).cpu().numpy()
                #
                #     _, ax = plt.subplots(2)
                #     ax[0].imshow(img)
                #     ax[1].imshow(heat_map_img)
                #     plt.show()
                #     config['batch_size'] = original_batch_size

            torch.cuda.empty_cache()
            iter += 1

if __name__ == '__main__':
    config = {'height': 214, 'width': 214, 'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1e7), 'z_dim': 32}

    train(config)