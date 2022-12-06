import multiprocessing
import os
import pickle
import sys
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import norm
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.optim import RMSprop
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, transforms, Resize, Normalize

sys.path.append('/home/zia/Desktop/MasterThesis/')

from thesis.training_result import TrainingResult, clean_tensor_str

from thesis.dataset import OneClassDataset, RandomPermutationTransform
from thesis.models import Encoder, Discriminator

# min max value for each class after applying resize and global contrastive normalization
CIFAR10_MIN_MAX = [[-25.502487182617188, 14.122282981872559], [-6.209507465362549, 8.61945629119873], [-30.79606056213379, 14.055601119995117], [-10.863265991210938, 10.729541778564453], [-9.717262268066406, 10.797706604003906], [-9.01405143737793, 9.231534004211426], [-7.824115753173828, 12.017197608947754], [-6.087644100189209, 11.072850227355957], [-15.263185501098633, 14.233451843261719], [-5.253917694091797, 8.3646240234375]]

def train(config):
    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])

    dataset = CIFAR10(root='../', train=True, download=True)
    normlization_transforms = transforms.Compose([ToTensor(), Resize((config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # normalizate_augmented_transform = transforms.Compose([ToTensor(), Resize((config['height'], config['width'])), RandomPermutationTransform(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=normlization_transforms)
    # augmented_inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=normalizate_augmented_transform)
    outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=normlization_transforms)

    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7*len(inlier_dataset))))
    # train_augmented_inlier_dataset = Subset(augmented_inlier_dataset, range(0, (int)(.7*len(inlier_dataset))))
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7*len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    train_dataset = train_inlier_dataset
    # train_augmented_dataset = train_augmented_inlier_dataset

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    f = Discriminator(config).cuda()
    e = Encoder(config).cuda()
    # e.load_state_dict(torch.load(f'model_autoencoder/run_1668195952/encoder_class_{config["class"]}'))
    f.train()
    e.train()
    f.apply(weights_init)
    e.apply(weights_init)

    optim_f = RMSprop(f.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    optim_e = RMSprop(e.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    scaling = config['var_scale']
    z_dist = MultivariateNormal(torch.zeros(config['z_dim']).cuda(), scaling*torch.eye(config['z_dim']).cuda())

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
    # discriminator_augmented_dataloader_iter = iter(DataLoader(train_augmented_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))

    def _next(iter):
        try:
            item = next(iter)
            return True, item
        except StopIteration:
            return False, None

    try:
        _, starting_roc, _ = evaluate(train_dataset, validation_dataset, e, config)
    except ValueError:
        starting_roc = torch.tensor([-1, -1, -1])
    training_result = TrainingResult(config, starting_roc)
    print(f'class: {config["class"]}, starting_roc: {starting_roc}')
    previous_mean = torch.zeros((config['z_dim'])).cuda()
    # last_epoch_em_distance = 0
    for encoder_iter in range(1, config['encoder_iters']+1):
        for _ in range(config['n_critic']):
            items_left, batch = _next(discriminator_dataloader_iter)
            if not items_left:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
                _, batch = _next(discriminator_dataloader_iter)
                # last_epoch_em_distance = 0

            # items_left, batch_aug = _next(discriminator_augmented_dataloader_iter)
            # if not items_left:
            #     discriminator_augmented_dataloader_iter = iter(DataLoader(train_augmented_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
            #     _, batch_aug = _next(discriminator_augmented_dataloader_iter)


            x, _ = batch
            x = x.cuda()
            # x_aug, _ = batch_aug
            # x_aug = x_aug.cuda()
            z = z_dist.sample((config['batch_size'],)).cuda()

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

        # items_left, batch_aug = _next(discriminator_augmented_dataloader_iter)
        # if not items_left:
        #     discriminator_augmented_dataloader_iter = iter(DataLoader(train_augmented_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20))
        #     _, batch_aug = _next(discriminator_augmented_dataloader_iter)

        (x, l) = batch
        x = x.cuda()
        # x_aug, _ = batch_aug
        # x_aug = x_aug.cuda()
        optim_e.zero_grad()
        e_x = e(x)
        loss = -torch.mean(f(e_x))
        loss.backward()
        optim_e.step()

        if (encoder_iter) % 100 == 0:
            try:
                cov, roc_auc, mean = evaluate(train_dataset, validation_dataset, e, config)
                eig_val, eig_vec = eig(cov)
                condition_no = torch.max(torch.real(eig_val)).item() / torch.min(torch.real(eig_val)).item()
            except ValueError:
                print(f'exception in class {config["class"]}')
                with open(f'{config["result_directory"]}/{config["dataset"]}_{config["class"]}_{config["instance"]}', 'wb') as file:
                    pickle.dump(training_result, file, protocol=pickle.HIGHEST_PROTOCOL)
                return
            training_result.update(cov, roc_auc, eig_val, eig_vec, e, previous_mean, mean)

            if encoder_iter % (4 * 100) == 0:
                print(f'mean:{torch.norm(mean).item()}\ncov: {cov}\n\n{encoder_iter}\n{training_result}\ncondition_no:{np.round(condition_no, 2)}\nroc:{clean_tensor_str(roc_auc)}'
                      f'\nmean_diff: {torch.norm(mean - previous_mean).item()}')
            previous_mean = mean


    with open(f'{config["result_directory"]}/{config["dataset"]}_{config["class"]}_{config["instance"]}', 'wb') as file:
        pickle.dump(training_result, file, protocol=pickle.HIGHEST_PROTOCOL)


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
        roc_auc = torch.tensor(roc_auc)

        e.train()

    return co_var, roc_auc, mean

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
    result_directory = f'model_cifar_20/run_28'

    config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'encoder_iters': (int)(10000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1, 'result_directory': result_directory}

    _class = (int)(sys.argv[1])
    _instance = (int)(sys.argv[2])

    config['class'] = _class
    config['instance'] = _instance
    train(config)