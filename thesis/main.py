import multiprocessing
import pickle
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import norm, kl_div
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.optim import RMSprop
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize
from tqdm import tqdm

import sys

sys.path.append('/home/zia/Desktop/MasterThesis/')

from thesis.training_result import TrainingResult


from thesis.dataset import OneClassDataset
from thesis.models import Encoder, Discriminator


def train(config):
    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])

    if config['dataset'] == 'cifar':
        dataset = CIFAR10(root='../', train=True, download=True)
        normalization_transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif config['dataset'] == 'mnist':
        dataset = MNIST(root='../', train=True, download=True)
        normalization_transform = Normalize((0.5), (0.5))

    inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), normalization_transform]))
    outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), normalization_transform]))

    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7*len(inlier_dataset))))
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7*len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    train_dataset = train_inlier_dataset

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

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))

    def _next(iter):
        try:
            item = next(iter)
            return True, item
        except StopIteration:
            return False, None

    starting_roc = evaluate(train_dataset, validation_dataset, e, config)
    training_result = TrainingResult(config, starting_roc)
    encoder_epoch = 0
    progress_bar = tqdm(range(1, config['encoder_iters']+1))
    for encoder_iter in progress_bar:
        for _ in range(config['n_critic']):
            items_left, batch = _next(discriminator_dataloader_iter)

            if not items_left:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
                _, batch = _next(discriminator_dataloader_iter)

            x, l = batch
            x = x.cuda()
            z = z_dist.sample_n(config['batch_size']).cuda()

            optim_f.zero_grad()
            loss = -(torch.mean(f(z)) - torch.mean(f(e(x))))
            loss.backward()
            optim_f.step()

            for p in f.parameters():
                p.data = p.data.clamp(-config['clip'], config['clip'])

        items_left, batch = _next(encoder_dataloader_iter)

        if not items_left:
            encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
            _, batch = _next(encoder_dataloader_iter)
            encoder_epoch += 1
            progress_bar.set_description(f'encoder_epoch: {encoder_epoch}')

        (x, l) = batch
        x = x.cuda()
        optim_e.zero_grad()
        e_x = e(x)
        loss = -torch.mean(f(e_x))
        loss.backward()
        optim_e.step()

        torch.cuda.empty_cache()

        if (encoder_iter) % 100 == 0:
            cov, var, roc_auc = evaluate(train_dataset, validation_dataset, e, config)
            eig_val, eig_vec = eig(cov)
            training_result.update(cov, var, roc_auc, eig_val, eig_vec, e)

            if encoder_iter % (4 * 100) == 0:
                print(training_result)

    with open(f'{config["dataset"]}_{config["class"]}_{time()}', 'wb') as file:
        pickle.dump(training_result, file, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate(train_dataset, validation_dataset, e, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
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
        co_var = torch.cov(zs.T, correction=0)
        dist = MultivariateNormal(loc=mean, covariance_matrix=co_var)
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
        roc_auc.append(roc_auc_score(np.abs(targets - 1), scores[2]))
        roc_auc = torch.tensor(roc_auc)

        var_samples = []
        for i in range(100):
            random_unit = torch.rand_like(zs[0])
            random_unit /= torch.norm(random_unit)

            projections = zs @ random_unit
            var_samples.append(torch.var(projections).item())

        var = torch.tensor(var_samples).mean().item()

        # print(f'iter: {iter}, roc: {roc_auc}, mean: {mean}, cov: {co_var}')

        e.train()

    return co_var, var, roc_auc

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
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'encoder_iters': (int)(20000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1}
    # config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1000), 'z_dim': 32, 'dataset': 'mnist', 'var_scale': 1}

    with NoDaemonProcessPool(processes=10) as pool:
        configs = []
        for i in range(0, 10):
            _config = config.copy()
            _config['class'] = i
            configs.append(_config)

        for _ in pool.imap_unordered(train, configs):
            pass