from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from torch.distributions import MultivariateNormal

from linearity_test_2.models import Encoder


def analyse():

    result_file = 'results/result_'

    with open(result_file, 'rb') as file:
        result = load(file)

    config = result.config
    distribution = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    projection = result.projection

    encoding_samples = distribution.sample((1000, ))
    prob = torch.exp(distribution.log_prob(encoding_samples)).numpy()
    data_samples = projection(encoding_samples)
    data_samples = data_samples.numpy()

    cmap = 'Greys'
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.scatter(xs=encoding_samples[:, 0], ys=encoding_samples[:, 1], zs=encoding_samples[:, 2], marker='.', c=prob, cmap=cmap)
    ax.scatter(xs=data_samples[:, 0], ys=data_samples[:, 1], zs=data_samples[:, 2], marker='.', c=prob, cmap='Reds')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(xs=data_samples[:, 0], ys=data_samples[:, 1], zs=data_samples[:, 2], marker='.', c=prob, cmap='Reds')
    start, end, step = -5, 5, 0.05
    x, y, z = torch.arange(start, end, step), torch.arange(start, end, step), torch.arange(start, end, step)
    # x, y, z = torch.arange(2, 4, step), torch.arange(0, 4, step), torch.arange(-6, 3, step)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    encoder = Encoder(result.config)
    encoder.load_state_dict(result.min_condition_no_model)
    encoder.eval()
    encoder.cuda()
    data_samples = torch.stack([grid_x, grid_y, grid_z]).reshape(3, -1).t().cuda()
    with torch.no_grad():
        encoding_samples = encoder(data_samples.to(torch.float))
    distribution = result.min_condition_no_distribution
    prob = []
    batch_size = 512000
    for i in range(batch_size, encoding_samples.shape[0]+1, batch_size):
        prob.append(torch.exp(distribution.log_prob(encoding_samples[i-batch_size:i])))

    if (encoding_samples.shape[0] % batch_size) != 0:
        prob.append(torch.exp(distribution.log_prob(encoding_samples[-(encoding_samples.shape[0] % batch_size):])))

    prob = torch.cat(prob)

    threashold = 0.005
    plot_samples = data_samples[prob.cpu() > threashold, :].cpu().numpy()
    plot_prob = prob[prob > threashold].cpu().numpy()
    ax.scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    pyplot.show()

