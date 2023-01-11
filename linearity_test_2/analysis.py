from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from torch import det
from torch.distributions import MultivariateNormal

from linearity_test_2.models import Encoder


def analyse(config):

    result_file = 'results/result_'

    # fig = pyplot.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # probs = []

    prob_sum = None
    prob_count = 0
    fig = pyplot.figure()
    for i, result_file in enumerate(listdir(config['result_folder'])):

        with open(config['result_folder']+result_file, 'rb') as file:
            result = load(file)

        config = result.config
        distribution = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
        projection = result.projection

        encoding_samples = distribution.sample((1000, ))
        prob = torch.exp(distribution.log_prob(encoding_samples)).numpy()
        data_samples = projection(encoding_samples)
        data_samples = data_samples.numpy()

        cmap = 'Greys'

        if i == 0:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            #
            ax.scatter(xs=encoding_samples[:, 0], ys=encoding_samples[:, 1], zs=encoding_samples[:, 2], marker='.', c=prob, cmap=cmap)
            ax.scatter(xs=data_samples[:, 0], ys=data_samples[:, 1], zs=data_samples[:, 2], marker='.', c=prob, cmap='Reds')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')


            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.scatter(xs=data_samples[:, 0], ys=data_samples[:, 1], zs=data_samples[:, 2], marker='.', c=prob, cmap='Reds')

        start, end, step = -10, 10, 0.5
        x, y, z = torch.arange(start, end, step), torch.arange(start, end, step), torch.arange(start, end, step)
        # x, y, z = torch.arange(-4, 4, step), torch.arange(1, 4, step), torch.arange(7, 10, step)
        # x, y, z = torch.arange(-3, -2.5, step), torch.arange(2, 2.5, step), torch.arange(8, 9, step)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        encoder = Encoder(result.config)
        encoder.load_state_dict(result.latest_model)
        encoder.eval()
        encoder.cuda()
        data_samples = torch.stack([grid_x, grid_y, grid_z]).reshape(3, -1).t().cuda()
        with torch.no_grad():
            encoding_samples = encoder(data_samples.to(torch.float))
        distribution = result.latest_distribution
        prob = []
        batch_size = 512000
        for i in range(batch_size, encoding_samples.shape[0]+1, batch_size):
            prob.append(torch.exp(distribution.log_prob(encoding_samples[i-batch_size:i])))

        if (encoding_samples.shape[0] % batch_size) != 0:
            prob.append(torch.exp(distribution.log_prob(encoding_samples[-(encoding_samples.shape[0] % batch_size):])))

        prob = torch.cat(prob)
        cov = distribution.covariance_matrix
        d = cov.shape[0]
        Z = torch.sqrt(torch.pow(torch.tensor([2*torch.pi]).cuda(), d) * det(cov))
        prob = prob * Z

        if torch.any(torch.isnan(prob)).item():
            continue

        assert torch.max(prob).item() <= 1.1, f'prob upper bound error, {torch.max(prob).item()}'
        assert torch.min(prob).item() >= 0, 'prob lower bound error'

        if prob_sum == None:
            prob_sum = prob
        else:
            prob_sum += prob

        prob_count += 1

        # threashold = 0.999999
        # plot_samples = data_samples[prob.cpu() > threashold, :].cpu().numpy()
        # plot_prob = prob[prob > threashold].cpu().numpy()

        # percentage = 1
        # sorted_index = torch.argsort(prob, descending=True)
        # indexes = sorted_index[:(int)(prob.shape[0] * percentage / 100)]
        # plot_prob = prob[indexes].cpu().numpy()
        # plot_samples = data_samples[indexes, :].cpu().numpy()


        # ax.scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob)
        #
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

        # probs.append(prob)

        # prob = torch.mean(torch.nan_to_num(torch.stack(probs), 0), dim=0)
        #

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count

    assert torch.max(prob).item() <= 1.0, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'

    percentage = 1
    sorted_index = torch.argsort(prob, descending=True)
    indexes = sorted_index[:(int)(prob.shape[0] * percentage / 100)]
    plot_prob = prob[indexes].cpu().numpy()
    plot_samples = data_samples[indexes, :].cpu().numpy()

    # threashold = 0.2
    # plot_samples = data_samples[prob.cpu() > threashold, :].cpu().numpy()
    # plot_prob = prob[prob > threashold].cpu().numpy()

    ax.scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    pyplot.show()

