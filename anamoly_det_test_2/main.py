from pickle import dumps, dump
from time import localtime, mktime, sleep

import torch.nn
from torch import softmax, sigmoid, nn
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from linearity_test_2.analysis import analyse
from linearity_test_2.models import Discriminator, Encoder, Projection
from linearity_test_2.datasets import ProjectedDataset
from linearity_test_2.result import training_result


def train_encoder(config):
    distribution = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    projection = Projection(config)

    train_dataset = ProjectedDataset(True, distribution, projection)
    validation_dataset = ProjectedDataset(True, distribution, projection)

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

    def _next(iter):
        try:
            batch = next(iter)
            return False, batch
        except StopIteration:
            return True, None

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True))
    optim_f = SGD(f.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optim_e = SGD(e.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    mean, cov, condition_no = evaluate_encoder(e, train_dataset, validation_dataset, config)
    result = training_result(projection, config)
    result.update(e, mean, cov, condition_no)
    result_file_name = f'results/set/result_{(int)(mktime(localtime()))}'
    # result_file_name = f'results/result_'

    for encoder_iter in range(1, config['encoder_iters']+1):

        for _ in range(config['discriminator_n']):
            empty, batch = _next(discriminator_dataloader_iter)
            if empty:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True))
                _, batch = _next(discriminator_dataloader_iter)

            x, _ = batch
            x = x.cuda()
            z = normal_dist.sample((x.shape[0], )).cuda()

            loss = -torch.mean(f(z) - f(e(x)))

            optim_f.zero_grad()
            loss.backward()
            optim_f.step()

            for parameter in f.parameters():
                parameter.data.clamp(-config['clip'], config['clip'])

        empty, batch = _next(encoder_dataloader_iter)
        if empty:
            encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True))
            _, batch = _next(encoder_dataloader_iter)

        x, _ = batch
        x = x.cuda()

        loss = -torch.mean(f(e(x)))

        optim_e.zero_grad()
        loss.backward()
        optim_e.step()

        if encoder_iter % 100 == 0:
            mean, cov, condition_no = evaluate_encoder(e, train_dataset, validation_dataset, config)
            result.update(e, mean, cov, condition_no)
            print(f'iter: {encoder_iter}, mean: {torch.norm(mean).item() : .4f}, condition_no: {condition_no.item(): .4f}')

    with open(result_file_name, 'wb') as file:
        dump(result, file)

def evaluate_encoder(encoder, train_dataset, validation_dataset, config):
    encoder.eval()

    with torch.no_grad():
        for dataset in [train_dataset]:
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
            encodings = []
            for x, _ in train_dataloader:
                x = x.cuda()
                encodings.append(encoder(x))

            encodings = torch.cat(encodings)
            mean = torch.mean(encodings, dim=0)
            # print(f'std: {torch.std(encodings, dim=0)}')
            cov = torch.cov(encodings.t(), correction=0)
            eig_val, eig_vec = eig(cov)
            condition_no = torch.max(eig_val.real) / torch.min(eig_val.real)

    encoder.train()

    return mean, cov, condition_no

if __name__ == '__main__':
    config = {'batch_size': 64, 'epochs': 200, 'data_dim': 3, 'encoding_dim': 3, 'encoder_iters': 0, 'discriminator_n': 4, 'lr': 1e-3, 'weight_decay': 1e-5, 'clip': 1e-2}

    config['class'] = 0
    # train_classifier(config)
    # train_binary_classifier(config)

    for _ in range(30):
        train_encoder(config)

    analyse()