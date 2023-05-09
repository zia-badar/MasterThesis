import sys
from os import mkdir, rmdir, listdir
from pickle import dumps, dump
from time import localtime, mktime, time
import shutil

import numpy as np
import torch.nn
from numpy import arange
from sklearn.metrics import roc_auc_score
from torch import softmax, sigmoid, nn, det
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, CosineSimilarity
from torch.optim import Adam, SGD, RMSprop
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Resize
from tqdm import tqdm

from anamoly_det_test_3.analysis import analyse
from anamoly_det_test_3.datasets import ProjectedDataset
from anamoly_det_test_3.models import Discriminator, Encoder, Projection
from anamoly_det_test_3.result import training_result


def train_encoder(config):
    f = Discriminator(config).cuda()
    e = Encoder(config).cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    f.apply(weights_init)
    e.apply(weights_init)

    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    train_dataset = ProjectedDataset(train=True, distribution=normal_dist, projection=Projection(config))

    def _next(iter):
        try:
            batch = next(iter)
            return False, batch
        except StopIteration:
            return True, None

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    optim_f = RMSprop(f.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optim_e = RMSprop(e.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    result = training_result(config)
    mean, cov, condition_no = evaluate_encoder(e, train_dataset, config)
    if config['encoding_dim'] == 1:
        cov = cov.unsqueeze(0).unsqueeze(0)
    result.update(e, mean, cov, condition_no)
    result_file_name = f'{config["result_folder"]}result_{(int)(time() * 1000)}'

    progress_bar = tqdm(range(1, config['encoder_iters']+1))
    for encoder_iter in progress_bar:

        for _ in range(config['discriminator_n']):
            empty, batch = _next(discriminator_dataloader_iter)
            if empty:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
                _, batch = _next(discriminator_dataloader_iter)

            x, _ = batch
            x = x.cuda()
            z = normal_dist.sample((x.shape[0], )).cuda()

            loss = -torch.mean(f(z) - f(e(x)))

            optim_f.zero_grad()
            loss.backward()
            optim_f.step()

            for parameter in f.parameters():
                parameter.data = parameter.data.clamp(-config['clip'], config['clip'])

        empty, batch = _next(encoder_dataloader_iter)
        if empty:
            encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
            _, batch = _next(encoder_dataloader_iter)

        x, _ = batch
        x = x.cuda()

        loss = -torch.mean(f(e(x)))

        optim_e.zero_grad()
        loss.backward()
        optim_e.step()
        progress_bar.set_description(f'loss: {loss.item()}')

        if encoder_iter % config['encoder_iters'] == 0:
            mean, cov, condition_no = evaluate_encoder(e, train_dataset, config)
            if config['encoding_dim'] == 1:
                cov = cov.unsqueeze(0).unsqueeze(0)
            result.update(e, mean, cov, condition_no)

    with open(result_file_name, 'wb') as file:
       dump(result, file)

def evaluate_encoder(encoder, train_dataset, config):
    encoder.eval()

    with torch.no_grad():
        for dataset in [train_dataset]:
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            encodings = []
            for x, _ in train_dataloader:
                x = x.cuda()
                encodings.append(encoder(x))

            encodings = torch.cat(encodings)
            mean = torch.mean(encodings, dim=0)
            cov = torch.cov(encodings.t(), correction=0)
            # eig_val, eig_vec = eig(cov)
            # condition_no = torch.max(eig_val.real) / torch.min(eig_val.real)
            condition_no = -1

    encoder.train()

    return mean, cov, condition_no

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = {'batch_size': 64, 'epochs': 200, 'encoding_dim': 2, 'projection_dim': 3, 'encoder_iters': 500, 'manifold_type': 'disconnected', 'discriminator_n': 5, 'lr': 5e-5, 'weight_decay': 1e-6, 'clip': 1e-2, 'num_workers': 0, 'result_folder': f'results/set_{(int)(time() * 1000)}/' }
    mkdir(config['result_folder'])

    for _ in range(10):
        train_encoder(config)
    analyse(config)
