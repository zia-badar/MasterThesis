import sys
from os import listdir
from pickle import load

import numpy as np
import torch
from matplotlib import pyplot
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from torch import det
from torch.nn.functional import normalize
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10

from anamoly_det_test_2.datasets import OneClassDataset, AugmentedDataset
from anamoly_det_test_2.models import Encoder, Model


def analyse(config):

    result_file = 'results/result_'

    # fig = pyplot.figure()
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # probs = []

    inlier_dataset = OneClassDataset(CIFAR10(root='../', train=True), one_class_labels=[config['class']])
    train_dataset = Subset(inlier_dataset, range(0, (int)(0.7 * len(inlier_dataset))))

    validation_inlier_dataset = Subset(inlier_dataset, range((int)(0.7 * len(inlier_dataset)), len(inlier_dataset)))
    outlier_classes = list(range(10))
    outlier_classes.remove(config['class'])
    outlier_dataset = OneClassDataset(CIFAR10(root='../', train=True), zero_class_labels=outlier_classes)
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])
    # validation_dataset = AugmentedDataset(validation_dataset, pair=False)
    without_pair_train_dataset = AugmentedDataset(train_dataset, pair=False)

    train_dataset = AugmentedDataset(train_dataset)

    # model = efficient_net(config).cuda()
    model = Model(3).cuda()
    model.load_state_dict(torch.load('../constrastive_model_512_128_8_layers_without_aug'))
    model.cuda()
    model.eval()

    # training_dataloader = DataLoader(without_pair_train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])
    #
    # train_x = []
    # with torch.no_grad():
    #     for x, l in training_dataloader:
    #         x, _ = model(x.cuda())
    #         train_x.append(normalize(x, dim=1))
    #         # train_x.append(x)
    # train_x = torch.cat(train_x)
    # gamma = (10. / (torch.var(train_x).item() * train_x.shape[1]))
    # svm = OneClassSVM(kernel='rbf', gamma=gamma).fit(train_x.cpu().numpy())
    # # svm = OneClassSVM(kernel='linear').fit(train_x.cpu().numpy())
    #
    # val_x = []
    # labels = []
    # with torch.no_grad():
    #     for x, l in validation_dataloader:
    #         x, _ = model(x.cuda())
    #         val_x.append(normalize(x, dim=1))
    #         # val_x.append(x)
    #         labels.append(l)
    # val_x = torch.cat(val_x).cpu().numpy()
    # labels = torch.cat(labels).cpu().numpy()
    #
    # score = svm.score_samples(val_x)
    # print(f'roc: {roc_auc_score(labels, score)}')
    # exit()

    datasets = [validation_inlier_dataset, outlier_dataset]
    colors = ['#000000', '#00FFFF']
    limit = 1
    contrastive_features_set = []
    with torch.no_grad():
        for dataset in datasets:
            contrastive_features = []
            dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
            for i, (x, l) in enumerate(dataloader):
                x, _ = model(x.cuda())
                contrastive_features.append(x)
                if i > limit:
                    break
            contrastive_features = torch.cat(contrastive_features).cpu().numpy()
            contrastive_features_set.append(contrastive_features)

    prob_sum = None
    prob_count = 0
    fig = pyplot.figure()
    for i, result_file in enumerate(listdir(config['result_folder'])):

        # if not result_file.endswith('_2000'):
        #     continue

        with open(config['result_folder'] + result_file, 'rb') as file:
            result = load(file)

        # if i == 0:
            # ax = fig.add_subplot(1, 2, 1, projection='3d')
            #
            # for i, contrastive_features in enumerate(contrastive_features_set):
            #     ax.scatter(xs=contrastive_features[:, 0], ys=contrastive_features[:, 1], zs=contrastive_features[:, 2], marker='.', c = colors[i])
            #
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')

            # ax = fig.add_subplot(1, 2, 2, projection='3d')
            # for i, contrastive_features in enumerate(contrastive_features_set):
            #     ax.scatter(xs=contrastive_features[:, 0], ys=contrastive_features[:, 1], zs=contrastive_features[:, 2], marker='.', c = colors[i])

        # if result.min_condition_no == sys.maxsize:
        #     continue
        # if result.min_condition_no > 100:
        #     continue
        if result.latest_condition_no < 0:
            continue

        # start, end, step = -10, 10, 0.5
        # x, y, z = torch.arange(start, end, step), torch.arange(start, end, step), torch.arange(start, end, step)
        # # x, y, z = torch.arange(-1, 1, step), torch.arange(-2, 2, step), torch.arange(-.25, .25, step)
        # grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
        encoder = Encoder(config)
        encoder.load_state_dict(result.min_condition_no_model)
        encoder.eval()
        encoder.cuda()
        # data_samples = torch.stack([grid_x, grid_y, grid_z]).reshape(3, -1).t().cuda()
        data_samples = []
        labels = []
        with torch.no_grad():
            for x, l in validation_dataloader:
                x, _ = model(x.cuda())
                data_samples.append(x)
                labels.append(l)
        data_samples = torch.cat(data_samples)
        labels = torch.cat(labels).cpu().numpy()


        with torch.no_grad():
            encoding_samples = encoder(data_samples)

        distribution = result.min_condition_no_distribution
        prob = []
        batch_size = 512000
        for i in range(batch_size, encoding_samples.shape[0]+1, batch_size):
            # prob.append(torch.exp(distribution.log_prob(encoding_samples[i-batch_size:i]).to(torch.float64)))
            log_prob = np.float128(distribution.log_prob(encoding_samples[i-batch_size:i]).cpu().numpy())
            prob.append(np.exp(log_prob))

        if (encoding_samples.shape[0] % batch_size) != 0:
            # prob.append(torch.exp(distribution.log_prob(encoding_samples[-(encoding_samples.shape[0] % batch_size):]).to(torch.float64)))
            log_prob = np.float128(distribution.log_prob(encoding_samples[-(encoding_samples.shape[0] % batch_size):]).cpu().numpy())
            prob.append(np.exp(log_prob))

        # prob = torch.cat(prob)
        prob = np.concatenate(prob)
        cov = distribution.covariance_matrix
        cov = cov.to(torch.float64)
        eig_val = torch.real(torch.linalg.eig(cov)[0]).to(torch.float64)
        d = cov.shape[0]
        # Z = torch.sqrt(torch.pow(torch.tensor([2*torch.pi], dtype=torch.float64).cuda(), d) * det(cov))
        # Z = np.sqrt((np.power(np.float128(2*np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()))))
        Z = np.sqrt((np.power(np.float128(2*np.pi), d) * np.prod(np.float128(eig_val.cpu().numpy()[:256])))) * np.sqrt(np.prod(np.float128(eig_val.cpu().numpy()[256:])))
        prob = prob * Z
        prob = torch.tensor(np.float64(prob))

        if torch.any(torch.isnan(prob)).item():
            continue

        assert torch.max(prob).item() <= 1.0, f'prob upper bound error, {torch.max(prob).item()}'
        assert torch.min(prob).item() >= 0, 'prob lower bound error'

        if prob_sum == None:
            prob_sum = prob
        else:
            prob_sum += prob

        prob_count += 1

        print(f'roc: {roc_auc_score(labels, (prob_sum/prob_count).cpu().numpy())}')
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
        # threashold = 1
        # plot_samples = data_samples[prob.cpu() > threashold, :].cpu().numpy()
        # plot_prob = prob[prob > threashold].cpu().numpy()
        # ax.scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob)

    print(f'not_nan: {prob_count}')
    prob = prob_sum / prob_count

    assert torch.max(prob).item() <= 1, 'prob upper bound error'
    assert torch.min(prob).item() >= 0, 'prob lower bound error'

    print(f'roc: {roc_auc_score(labels, prob.cpu().numpy())}')
    exit()
    percentage = 100
    sorted_index = torch.argsort(prob, descending=True)
    indexes = sorted_index[:(int)(prob.shape[0] * percentage / 100)]
    plot_prob = prob[indexes].cpu().numpy()
    plot_samples = data_samples[indexes, :].cpu().numpy()

    # threashold = 0.69
    # plot_samples = data_samples[prob.cpu() > threashold, :].cpu().numpy()
    # plot_prob = prob[prob > threashold].cpu().numpy()

    ax.scatter(xs=plot_samples[:, 0], ys=plot_samples[:, 1], zs=plot_samples[:, 2], marker='.', c=plot_prob, cmap='spring')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    pyplot.show()

