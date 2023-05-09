from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from torch import det
from torch.distributions import MultivariateNormal
from torch.nn import CosineSimilarity
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor

from anamoly_det_test_3.datasets import ProjectedDataset
from anamoly_det_test_3.models import Encoder, Projection


def analyse(config):
    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    train_dataset = ProjectedDataset(train=True, distribution=normal_dist, projection=Projection(config))

    start, end, step = -1, 1, 0.01

    if config['projection_dim'] == 3:
        x, y, z = torch.arange(0, 1, step), torch.arange(0.3, 0.6, step), torch.arange(-1, 0.5, step)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        grid_samples = torch.stack([grid_x, grid_y, grid_z]).reshape(3, -1).t().cuda()
    elif config['projection_dim'] == 2:
        x, y = torch.arange(-1, 2, step), torch.arange(-2, 0, step)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_samples = torch.stack([grid_x, grid_y]).reshape(2, -1).t().cuda()

    prob_sum = None
    prob_count = 0
    for i, result_file in enumerate(listdir(config['result_folder'])):

        with open(config['result_folder'] + result_file, 'rb') as file:
            result = load(file)

        model = Encoder(config)
        model.load_state_dict(result.latest_model)
        model.eval()
        model = model.cuda()

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        train_projection = []
        with torch.no_grad():
            for x, _ in train_dataloader:
                train_projection.append(x)

        train_projection = torch.cat(train_projection)
        # mean = torch.mean(encodings, dim=0)
        # cov = torch.cov(encodings.t(), correction=0)
        # distribution = MultivariateNormal(mean, cov)
        distribution = result.latest_distribution

        prob = []
        with torch.no_grad():
            grid_encoded_samples = model(grid_samples)
            log_prob = np.float128(distribution.log_prob(grid_encoded_samples).cpu().numpy())
            prob.append(np.exp(log_prob))

        prob = np.concatenate(prob)

        cov = distribution.covariance_matrix
        d = cov.shape[0]
        eig_val = torch.real(torch.linalg.eig(cov)[0]).to(torch.float64)
        # Z = torch.sqrt(torch.pow(torch.tensor([2*torch.pi], dtype=torch.float64).cuda(), d) * det(cov)).type(torch.float32)
        Z = np.sqrt((np.power(np.float128(2 * np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()))))
        prob = prob * Z
        prob = torch.tensor(np.float64(prob))

        if torch.any(torch.isnan(prob)).item():
            continue

        assert torch.max(prob).item() <= 1.1, f'prob upper bound error, {torch.max(prob).item()}'
        assert torch.min(prob).item() >= 0, 'prob lower bound error'

        if prob_sum == None:
            prob_sum = prob
        else:
            prob_sum += prob

        prob_count += 1
        # if (i+1) % 10 == 0:
        # print(f'{result_file}, roc_score: {roc_auc_score(labels.cpu().numpy(), (prob_sum / prob_count).cpu().numpy())}, roc: {roc_auc_score(labels.cpu().numpy(), prob.cpu().numpy())}')

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count

    assert torch.max(prob).item() <= 1.1, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'

    fig = pyplot.figure()
    if config['projection_dim'] == 3:
        ax = fig.subplots(1, 2, subplot_kw={'projection': '3d'})
    elif config['projection_dim'] == 2:
        ax = fig.subplots(1, 2)

    if config['projection_dim'] == 3:
        ax[0].scatter(xs=train_projection[:, 0].cpu().numpy(), ys = train_projection[:, 1].cpu().numpy(), zs = train_projection[:, 2].cpu().numpy())
        percentage = 1
    elif config['projection_dim'] == 2:
        ax[0].scatter(x=train_projection[:, 0].cpu().numpy(), y = train_projection[:, 1].cpu().numpy())
        percentage = 8

    sorted_index = torch.argsort(prob, descending=True)
    indexes = sorted_index[:(int)(prob.shape[0] * percentage / 100)]
    plot_prob = prob[indexes].cpu().numpy()
    plot_samples = grid_samples[indexes, :].cpu().numpy()

    if config['projection_dim'] == 3:
        ax[1].scatter(xs=train_projection[:, 0].cpu().numpy(), ys = train_projection[:, 1].cpu().numpy(), zs = train_projection[:, 2].cpu().numpy(), marker='.', c='r')
        ax[1].scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob, cmap='spring')
    elif config['projection_dim'] == 2:
        ax[1].scatter(x=train_projection[:, 0].cpu().numpy(), y = train_projection[:, 1].cpu().numpy(), marker='.', c='r')
        ax[1].scatter(x=plot_samples[:, 0], y=plot_samples[:, 1], marker='.', c=plot_prob, cmap='spring')

    for _ax in [ax[0], ax[1]]:
        _ax.set_xlabel('x')
        _ax.set_ylabel('y')
        if config['projection_dim'] == 3:
            _ax.set_zlabel('z')

    pyplot.show()
