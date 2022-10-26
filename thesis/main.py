import multiprocessing

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import norm
from torch.distributions import MultivariateNormal
from torch.optim import RMSprop
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize
from tqdm import tqdm

from thesis.dataset import OneClassDataset
from thesis.models import Encoder, Discriminator


def train(config):
    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])

    if config['dataset'] == 'cifar':
        dataset = OneClassDataset(CIFAR10(root='../', train=True, download=True), one_class_labels=inlier, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif config['dataset'] == 'mnist':
        dataset = OneClassDataset(MNIST(root='../', train=True, download=True), one_class_labels=inlier, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5), (0.5))]))

    split_size = (int)(.7*len(dataset))
    dataset_splits = random_split(dataset, [split_size, len(dataset) - split_size])
    dataset = dataset_splits[0]

    if config['dataset'] == 'cifar':
        _dataset = OneClassDataset(CIFAR10(root='../', train=True, download=True), zero_class_labels=outlier, transform=transforms.Compose([ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif config['dataset'] == 'mnist':
        _dataset = OneClassDataset(MNIST(root='../', train=True, download=True), zero_class_labels=outlier, transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), Normalize((0.5), (0.5))]))

    val_dataset = ConcatDataset([_dataset, dataset_splits[1]])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
    trn_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)

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

    optim_f = RMSprop(f.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    optim_e = RMSprop(e.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    scaling = config['var_scale']
    z_dist = MultivariateNormal(torch.zeros(config['z_dim']).cuda(), scaling*torch.eye(config['z_dim']).cuda())

    def pretty_tensor(tensor):
        return np.array_repr(tensor.detach().cpu().numpy()).replace('\n', '')
    resize = Resize(size=(config['height'], config['width']))

    # print(f'roc_auc: {evaluate(trn_dataloader, val_dataloader, e, config)}')

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

            if iter % (config['n_critic'] + 1) == 0:
                optim_e.zero_grad()
                e_x = e(x)
                loss = -torch.mean(f(e_x))
                loss.backward()
                optim_e.step()

                # a = torch.round(torch.cov(e_x.T, correction=0), decimals=4)
                # b = (torch.sum(torch.abs(a)) - torch.sum(torch.abs(torch.diag(a)))).item()
                # train_progress_bar.set_description(f'encoding_mean: {pretty_tensor(torch.round(torch.mean(e_x, dim=0), decimals=4))}, encoding_variance: {pretty_tensor(torch.round(torch.var(e_x, dim=0, unbiased=False), decimals=4))}, encoding co-variance: {pretty_tensor(a)}, {b}')

                # if iter % ((config['n_critic'] + 1)*10*10) == 0:
                #     roc_auc = evaluate(trn_dataloader, val_dataloader, e, config)
                #     print(f'roc_auc: {roc_auc}')

            torch.cuda.empty_cache()
            iter += 1

    roc_auc = evaluate(trn_dataloader, val_dataloader, e, config)
    torch.save(e.state_dict(), f'model_{config["dataset"]}_{config["class"]}')

    with open('output_results.log', 'a') as file:
        file.write(f'class: {config["class"]}, dataset: {config["dataset"]}, roc_auc: {roc_auc}\n')

def evaluate(train_dataloader, validation_dataloader, e, config):
    with torch.no_grad():
        e.eval()
        targets = []
        scores = [[], [], []]
        zs = []
        for _x, _l in train_dataloader:
            _x = _x.cuda()
            zs.append(e(_x))

        zs = torch.cat(zs)
        mean = torch.mean(zs, dim=0)
        var = torch.cov(zs.T, correction=0)
        dist = MultivariateNormal(loc=mean, covariance_matrix=var)
        scaling = config['var_scale']
        id_dist = MultivariateNormal(loc=torch.zeros(mean.shape[0]).cuda(), covariance_matrix=scaling*torch.eye(mean.shape[0]).cuda())

        for _x, _l in validation_dataloader:
            _x = _x.cuda()
            z = e(_x)
            scores[0].append(dist.log_prob(z))
            scores[1].append(id_dist.log_prob(z))
            scores[2].append(norm(z, dim=1))
            targets.append(_l)

        scores[0] = torch.cat(scores[0]).cpu().numpy()
        scores[1] = torch.cat(scores[1]).cpu().numpy()
        scores[2] = torch.cat(scores[2]).cpu().numpy()
        targets = torch.cat(targets).numpy()

        roc_auc = []
        roc_auc.append(roc_auc_score(targets, scores[0]))
        roc_auc.append(roc_auc_score(targets, scores[1]))
        roc_auc.append(roc_auc_score(targets, scores[2]))

        print(f'iter: {iter}, roc: {roc_auc}, mean: {mean}, cov: {var}')

        e.train()

    return roc_auc

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

if __name__ == '__main__':
    config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1000), 'z_dim': 128, 'dataset': 'cifar', 'var_scale': 1}
    # config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1000), 'z_dim': 32, 'dataset': 'mnist', 'var_scale': 1}
    with NoDaemonProcessPool(processes=10) as pool:
        configs = []
        for i in range(10):
            _config = config.copy()
            _config['class'] = i
            configs.append(_config)

        for _ in pool.imap_unordered(train, configs):
            pass
