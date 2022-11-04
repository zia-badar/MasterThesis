import multiprocessing

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import norm, kl_div
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import KLDivLoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize
from tqdm import tqdm

import sys
sys.path.append('/home/zia/Desktop/MasterThesis/')

from thesis.dataset import OneClassDataset
from thesis.models import Encoder, Discriminator
from thesis.normality import get_variance


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
    # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
    # validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)

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

    starting_roc = evaluate(train_dataset, validation_dataset, e, config)
    print(f'roc_auc: class: {config["class"]}, {starting_roc}')
    best_var = 1000
    best_var_roc = None
    best_roc = [0, 0, 0]

    col_epoch = -1
    max_var = 0
    max_eig_val = -1
    max_eig_val_epoch = -1
    min_eig_val = 1000
    min_eig_val_epoch = -1

    max_dit = -1
    min_dit = 1000
    min_dit_roc = None
    min_dit_epoch = -1
    max_dit_epoch = -1


    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))

    def _next(iter):
        try:
            item = next(iter)
            return True, item
        except StopIteration:
            return False, None

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
            # var = get_variance(trn_dataloader, e)
            if var < best_var:
                best_var = var
                best_var_roc = roc_auc
                best_epoch = encoder_epoch

            if var > 1:
                col_epoch = encoder_epoch
            if var > max_var:
                max_var = var

            best_roc = [max(best_roc[0], roc_auc[0]), max(best_roc[1], roc_auc[1]), max(best_roc[2], roc_auc[2])]

            L, V = eig(cov)
            max_eig = torch.max(torch.real(L)).item()
            min_eig = torch.min(torch.real(L)).item()
            if max_eig_val < max_eig:
                max_eig_val = max_eig
                max_eig_val_epoch = encoder_epoch

            if min_eig_val > min_eig and encoder_epoch > 30:
                min_eig_val = min_eig
                min_eig_val_epoch = encoder_epoch

            dit = torch.prod(torch.real(L)).item()

            if dit < min_dit:
                min_dit = dit
                min_dit_epoch = encoder_epoch
                min_dit_roc = roc_auc

            if dit > max_dit:
                max_dit = dit
                max_dit_epoch = encoder_epoch

            print(f'eig value: {L}\n, eig vector: {V}')
            print(f'eig_min: {min_eig_val}, epoch: {min_eig_val_epoch}')
            print(f'eig_max: {max_eig_val}, epoch: {max_eig_val_epoch}')
            print(f'dit_min: {min_dit}, roc: {min_dit_roc}, epoch: {min_dit_epoch}')
            print(f'dit_max: {max_dit}, epoch: {max_dit_epoch}')
            print(f'var: {var}, dit: {dit}, roc: {roc_auc}')
            print(f'min_var: {best_var}, min_var_roc: {best_var_roc}, best_epoch: {best_epoch}')
            print(f'max_var: {max_var}')
            print(f'best_roc: {best_roc}')
            print(f'col_epoch: {col_epoch}')

            # print(f'roc: {roc_auc}, var: {var}')
            # with open('output_results.log', 'a') as file:
            #     file.write(f'class: {config["class"]}, dataset: {config["dataset"]}, starting_roc_auc: {starting_roc} roc_auc: {roc_auc}\n')
            # torch.save(e.state_dict(), f'model_{config["dataset"]}_{config["class"]}_{epoch}')

def evaluate(train_dataset, validation_dataset, e, config):
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
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

        var_samples = []
        for i in range(100):
            random_unit = torch.rand_like(zs[0])
            random_unit /= torch.norm(random_unit)

            projections = zs @ random_unit
            var_samples.append(torch.var(projections).item())

        var = torch.tensor(var_samples).mean().item()

        print(f'iter: {iter}, roc: {roc_auc}, mean: {mean}, cov: {co_var}')

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

    _class = (int)(sys.argv[1])

    config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'encoder_iters': (int)(100000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1}
    # config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 6, 'clip': 1e-2, 'learning_rate': 5e-5, 'epochs': (int)(1000), 'z_dim': 32, 'dataset': 'mnist', 'var_scale': 1}

    config['class'] = _class
    train(config)

    # with NoDaemonProcessPool(processes=10) as pool:
    #     configs = []
    #     for i in range(5, 10):
    #         _config = config.copy()
    #         _config['class'] = i
    #         configs.append(_config)
    #
    #     for _ in pool.imap_unordered(train, configs):
    #         pass