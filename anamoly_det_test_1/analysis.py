from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from torch import det
from torch.distributions import MultivariateNormal
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from anamoly_det_test_1.datasets import OneClassDataset
from anamoly_det_test_1.models import Encoder


def analyse(config):

    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])
    dataset = CIFAR10(root='../', train=True, download=True)
    transform = transforms.Compose([ToTensor()])
    inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transform)
    outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transform)
    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    train_dataset = train_inlier_dataset
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    prob_sum = None
    prob_count = 0
    for i, result_file in enumerate(listdir(config['result_folder'])):

        with open(config['result_folder'] + result_file, 'rb') as file:
            result = load(file)

        # config = result.config
        # distribution = result.min_condition_no_distribution

        if not result_file.endswith('_1000'):
            continue


        validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        model = Encoder(config)
        model.load_state_dict(result.min_condition_no_model)
        model.eval()
        model = model.cuda()

        # train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        # encodings = []
        # with torch.no_grad():
        #     for x, _ in train_dataloader:
        #         x = x.cuda()
        #         encodings.append(model(x))
        #
        # encodings = torch.cat(encodings)
        # mean = torch.mean(encodings, dim=0)
        # cov = torch.cov(encodings.t(), correction=0)
        # distribution = MultivariateNormal(mean, cov)
        distribution = result.min_condition_no_distribution

        prob = []
        labels = []
        with torch.no_grad():
            for x, l in validation_dataloader:
                x = x.cuda()
                prob.append(torch.exp(distribution.log_prob(model(x))))
                labels.append(l)

        prob = torch.cat(prob)
        labels = torch.cat(labels)

        cov = distribution.covariance_matrix
        d = cov.shape[0]
        Z = torch.sqrt(torch.pow(torch.tensor([2*torch.pi], dtype=torch.float64).cuda(), d) * det(cov)).type(torch.float32)
        prob = prob * Z

        if torch.any(torch.isnan(prob)).item():
            continue

        assert torch.max(prob).item() <= 1, f'prob upper bound error, {torch.max(prob).item()}'
        assert torch.min(prob).item() >= 0, 'prob lower bound error'

        if prob_sum == None:
            prob_sum = prob
        else:
            prob_sum += prob


        prob_count += 1

        print(f'roc_score: {roc_auc_score(labels.cpu().numpy(), (prob_sum / prob_count).cpu().numpy())}')

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count

    assert torch.max(prob).item() <= 1, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'


    scores = prob.cpu().numpy()
    targets = labels.cpu().numpy()

    print(f'class: {config["class"]}, roc_score: {roc_auc_score(targets, scores)}, not_nan: {prob_count}')
