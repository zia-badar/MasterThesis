import csv
import pickle
from time import time

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize, transforms, ToTensor, Resize

from thesis.dataset import OneClassDataset
from thesis.main import evaluate
from thesis.models import Encoder


if __name__ == "__main__":
    folder_name = 'model_cifar_20/run_4/'

    classes = list(range(10))
    plt.rcParams['font.size'] = '12'
    fig, ax = plt.subplots(len(classes), 2, figsize=(10, len(classes)*5))
    spaces = ' ' * 30
    fig.suptitle('before training nomaly'+ spaces + 'before training anomaly'+spaces+'after training nomaly'+spaces+'after training anomlay')
    analysis_csv_name = 'analysis.csv'

    with open(analysis_csv_name, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class', 'p(z|u,c)', 'p(z|0,1)', '||z||', 'p(z|u,c)', 'p(z|0,1)', '||z||'])

    before_training_sum_roc = torch.zeros((3))
    after_training_sum_roc = torch.zeros((3))
    for axis_index, _class in enumerate(classes):
        config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'encoder_iters': (int)(10000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1, 'timestamp': (int)(time())}
        config['class'] = _class

        inlier = [config['class']]
        outlier = list(range(10))
        outlier.remove(config['class'])

        dataset = CIFAR10(root='../', train=True, download=True)
        normalization_transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), normalization_transform]))
        outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=transforms.Compose( [ToTensor(), Resize(size=(config['height'], config['width'])), normalization_transform]))

        train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7*len(inlier_dataset))))
        validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7*len(inlier_dataset)), len(inlier_dataset)))
        validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

        train_dataset = train_inlier_dataset

        without_training_encoder = Encoder(config).cuda()
        e = Encoder(config)
        e = e.cuda()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        e.apply(weights_init)

        _, _, before_training_roc = evaluate(train_dataset, validation_dataset, e, config)
        before_training_sum_roc += before_training_roc
        before_training_roc = list(np.round(before_training_roc.cpu().numpy(), 2))

        with open(folder_name + f'cifar_{config["class"]}', 'rb') as file:
            training_result = pickle.load(file)

        e.load_state_dict(training_result.min_condition_no_model)


        _, _, after_training_roc = evaluate(train_dataset, validation_dataset, e, config)
        after_training_sum_roc += after_training_roc
        after_training_roc = list(np.round(after_training_roc.cpu().numpy(), 2))

        with torch.no_grad():
            e.eval()
            random_vector = torch.rand(config['z_dim']).cuda()
            random_unit_vector = random_vector / torch.norm(random_vector)

            nominal_dataloader = DataLoader(validation_inlier_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
            anominal_dataloader = DataLoader(outlier_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
            projections = []
            for i, encoder in enumerate([without_training_encoder, e]):
                ax[axis_index, i].title.set_text(str(before_training_roc) if i == 0 else str(after_training_roc))
                if i == 0:
                    ax[axis_index, i].set_ylabel(f'class: {_class}')
                for j, dataloader in enumerate([nominal_dataloader, anominal_dataloader]):
                    projections = []

                    for (x, l) in dataloader:
                        x = x.cuda()
                        z = encoder(x)
                        projection = z @ random_unit_vector
                        projections.append(projection)

                    projections = torch.cat(projections)

                    ax[axis_index, i].hist(projections.cpu().numpy(), bins=100, density=True, histtype='step', color=('g' if j == 0 else 'r'))

        with open(analysis_csv_name, 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([_class] + before_training_roc + after_training_roc)
            if axis_index == len(classes) - 1:
                before_training_average_roc = before_training_sum_roc / len(classes)
                after_training_average_roc = after_training_sum_roc / len(classes)
                before_training_average_roc = list(np.round(before_training_average_roc.cpu().numpy(), 2))
                after_training_average_roc = list(np.round(after_training_average_roc.cpu().numpy(), 2))
                writer.writerow(['average'] + before_training_average_roc + after_training_average_roc)


    plt.show()




        # print(before_training_roc, after_training_roc, training_result.eig_max)

