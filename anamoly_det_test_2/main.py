import sys
from os import mkdir
from pickle import dumps, dump
from time import localtime, mktime, sleep

import numpy as np
import torch.nn
from sklearn.metrics import roc_auc_score
from torch import softmax, sigmoid, nn
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from anamoly_det_test_2.analysis import analyse
from anamoly_det_test_2.datasets import OneClassDataset, AugmentedDataset
from anamoly_det_test_2.models import Discriminator, Encoder, Model
from anamoly_det_test_2.result import training_result




def train_encoder(config):
    inlier_dataset = OneClassDataset(CIFAR10(root='../', train=True), one_class_labels=[config['class']])
    train_dataset = Subset(inlier_dataset, range(0, (int)(0.7 * len(inlier_dataset))))

    validation_inlier_dataset = Subset(inlier_dataset, range((int)(0.7 * len(inlier_dataset)), len(inlier_dataset)))
    outlier_classes = list(range(10))
    outlier_classes.remove(config['class'])
    outlier_dataset = OneClassDataset(CIFAR10(root='../', train=True), zero_class_labels=outlier_classes)
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])
    # validation_dataset = AugmentedDataset(validation_dataset, pair=False)
    # without_pair_train_dataset = AugmentedDataset(train_dataset, pair=False)
    without_pair_train_dataset = train_dataset

    # train_dataset = AugmentedDataset(train_dataset)

    # model = efficient_net(config).cuda()
    model = Model(3).cuda()
    model.load_state_dict(torch.load('../constrastive_model_512_128_8_layers_without_aug'))
    model.cuda()
    model.eval()

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

    discriminator_dataloader_iter = iter(DataLoader(without_pair_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    encoder_dataloader_iter = iter(DataLoader(without_pair_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    optim_f = SGD(f.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optim_e = SGD(e.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    # mean, cov, condition_no = evaluate_encoder(model, e, without_pair_train_dataset, validation_dataset, config)
    # print(f'iter: 0, mean: {torch.norm(mean).item() : .4f}, condition_no: {condition_no.item(): .4f}')
    result = training_result()
    # result.update(e, mean, cov, sys.maxsize-1)
    result_file_name = f'{config["result_folder"]}result_{(int)(mktime(localtime()))}'
    # result_file_name = f'results/result_'

    for encoder_iter in range(1, config['encoder_iters']+1):

        for _ in range(config['discriminator_n']):
            empty, batch = _next(discriminator_dataloader_iter)
            if empty:
                discriminator_dataloader_iter = iter(DataLoader(without_pair_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
                _, batch = _next(discriminator_dataloader_iter)

            x, _ = batch
            x, _ = model(x.cuda())
            z = normal_dist.sample((x.shape[0], )).cuda()

            loss = -torch.mean(f(z) - f(e(x)))

            optim_f.zero_grad()
            loss.backward()
            optim_f.step()

            for parameter in f.parameters():
                parameter.data.clamp(-config['clip'], config['clip'])

        empty, batch = _next(encoder_dataloader_iter)
        if empty:
            encoder_dataloader_iter = iter(DataLoader(without_pair_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
            _, batch = _next(encoder_dataloader_iter)

        x, _ = batch
        x, _ = model(x.cuda())

        loss = -torch.mean(f(e(x)))

        optim_e.zero_grad()
        loss.backward()
        optim_e.step()

        if encoder_iter % 100 == 0:
            mean, cov, condition_no = evaluate_encoder(model, e, without_pair_train_dataset, validation_dataset, config, encoder_iter == config['encoder_iters'] or encoder_iter == 100)
            result.update(e, mean, cov, condition_no)
            print(f'iter: {encoder_iter}, mean: {torch.norm(mean).item() : .4f}, condition_no: {condition_no.item(): .4f}')

    with open(result_file_name, 'wb') as file:
        dump(result, file)

def evaluate_encoder(model, encoder, train_dataset, validation_dataset, config, compute_roc=False):
    encoder.eval()

    with torch.no_grad():
        for dataset in [train_dataset]:
            train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            encodings = []
            for x, _ in train_dataloader:
                x, _ = model(x.cuda())
                encodings.append(encoder(x))

            encodings = torch.cat(encodings)
            mean = torch.mean(encodings, dim=0)
            # print(f'std: {torch.std(encodings, dim=0)}')
            cov = torch.cov(encodings.t(), correction=0)
            eig_val, eig_vec = eig(cov)
            condition_no = torch.max(eig_val.real) / torch.min(eig_val.real)

        if compute_roc:
            validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
            data_samples = []
            labels = []
            for x, l in validation_dataloader:
                x, _ = model(x.cuda())
                data_samples.append(x)
                labels.append(l)
            data_samples = torch.cat(data_samples)
            labels = torch.cat(labels).cpu().numpy()

            with torch.no_grad():
                encoding_samples = encoder(data_samples)

            distribution = MultivariateNormal(mean, cov)

            prob = []
            log_prob = np.float128( distribution.log_prob(encoding_samples).cpu().numpy())
            prob.append(np.exp(log_prob))

            # prob = torch.cat(prob)
            prob = np.concatenate(prob)
            cov = distribution.covariance_matrix
            cov = cov.to(torch.float64)
            eig_val = torch.real(torch.linalg.eig(cov)[0]).to(torch.float64)
            d = cov.shape[0]
            # Z = torch.sqrt(torch.pow(torch.tensor([2*torch.pi], dtype=torch.float64).cuda(), d) * det(cov))
            Z = np.sqrt((np.power(np.float128(2 * np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()))))
            # Z = np.sqrt((np.power(np.float128(2*np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()[:256])))) * np.sqrt(np.prod(np.float128(eig_val.cpu().numpy()[256:])))
            prob = prob * Z
            # prob = torch.tensor(np.float64(prob))
            print(f'roc: {roc_auc_score(labels, prob)}')

    encoder.train()

    return mean, cov, condition_no

if __name__ == '__main__':
    _class = 1
    config = {'batch_size': 64, 'epochs': 200, 'data_dim': 512, 'encoding_dim': 128, 'encoder_iters': 1000, 'discriminator_n': 4, 'lr': 1e-3, 'weight_decay': 1e-5, 'clip': 1e-2, 'num_workers': 20, 'result_folder': f'results/set_{(int)(mktime(localtime()))}_{_class}/' }

    config['class'] = _class
    mkdir(config['result_folder'])
    # train_classifier(config)
    # train_binary_classifier(config)

    for _ in range(10):
        try:
            train_encoder(config)
        except:
            print('exception')

    analyse(config)