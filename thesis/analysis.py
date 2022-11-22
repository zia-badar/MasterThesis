import csv
import pickle
from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Normalize, transforms, ToTensor, Resize

from thesis.dataset import OneClassDataset, GlobalContrastiveNormalizationTransform, MinMaxNormalizationTransform, \
    Random90RotationTransform, RandomPermutationTransform
from thesis.main import evaluate, CIFAR10_MIN_MAX
from thesis.models import Encoder


def area_under_curve(l):
    values = torch.tensor(l)
    result = torch.sum(0.5 * torch.abs(values[1:] - values[:-1])) + torch.sum(torch.min(torch.stack([values[1:], values[:-1]]), dim=0).values)
    return np.round(result.item(), 2)

def generate_plots_and_csv():
    folder_name = 'model_cifar_20/run_28/best_cond_no/'
    classes = list(range(10))
    classes = [c for c in classes if c not in []]
    # instances = list(range(5))
    instances = list(range(1))
    plt.rcParams['font.size'] = '12'
    fig, ax = plt.subplots(len(classes) * len(instances), 5, figsize=(25, len(classes)*len(instances)*5))
    analysis_csv_name = 'analysis.csv'

    with open(analysis_csv_name, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class', 'p(z|u,c)', 'p(z|0,1)', '||z||', 'p(z|u,c)', 'p(z|0,1)', '||z||'])

    before_training_sum_roc = torch.zeros((3))
    after_training_sum_roc = torch.zeros((3))
    for class_i, _class in enumerate(classes):
        for instance in instances:
            axis_index = class_i * len(instances) + instance
            config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5, 'encoder_iters': (int)(10000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1, 'timestamp': (int)(time())}
            config['class'] = _class

            inlier = [config['class']]
            outlier = list(range(10))
            outlier.remove(config['class'])

            dataset = CIFAR10(root='../', train=True, download=True)
            normlization_transforms = transforms.Compose( [ToTensor(), Resize((config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            normalizate_augmented_transform = transforms.Compose( [ToTensor(), Resize((config['height'], config['width'])), RandomPermutationTransform(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=normlization_transforms)
            augmented_inlier_dataset = OneClassDataset(dataset, one_class_labels=inlier, transform=normalizate_augmented_transform)
            outlier_dataset = OneClassDataset(dataset, zero_class_labels=outlier, transform=normlization_transforms)

            train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7*len(inlier_dataset))))
            validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7*len(inlier_dataset)), len(inlier_dataset)))
            validation_augmented_inlier_dataset = Subset(augmented_inlier_dataset, range((int)(.7*len(inlier_dataset)), len(inlier_dataset)))
            validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

            train_dataset = train_inlier_dataset

            without_training_encoder = Encoder(config).cuda()
            def weights_init(m):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    m.weight.data.normal_(0.0, 0.02)
                elif classname.find('BatchNorm') != -1:
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)
            without_training_encoder.apply(weights_init)

            _, before_training_roc, before_mean = evaluate(train_dataset, validation_dataset, without_training_encoder, config)
            before_training_sum_roc += before_training_roc
            before_training_roc = list(np.round(before_training_roc.cpu().numpy(), 2))

            # with open(folder_name + f'cifar_{config["class"]}_{instance}', 'rb') as file:
            with open(folder_name + f'cifar_{config["class"]}', 'rb') as file:
                training_result = pickle.load(file)

            trained_encoder = Encoder(config).cuda()
            trained_encoder.load_state_dict(training_result.min_condition_no_model)


            _, after_training_roc, after_mean = evaluate(train_dataset, validation_dataset, trained_encoder, config)
            after_training_sum_roc += after_training_roc
            after_training_roc = list(np.round(after_training_roc.cpu().numpy(), 2))

            with torch.no_grad():
                without_training_encoder.eval()
                trained_encoder.eval()
                random_vector = torch.rand(config['z_dim']).cuda()
                random_unit_vector = random_vector / torch.norm(random_vector)

                nominal_dataloader = DataLoader(validation_inlier_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
                augmented_nominal_dataloader = DataLoader(validation_augmented_inlier_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
                anominal_dataloader = DataLoader(outlier_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=20)
                for i, encoder in enumerate([without_training_encoder, trained_encoder]):
                    ax[axis_index, i].title.set_text('before: ' + str(before_training_roc) if i == 0 else 'after: ' + str(after_training_roc) + f', min_cond_no: {np.round(training_result.min_condition_no, 2)}')
                    if i == 0:
                        ax[axis_index, i].set_ylabel(f'class: {config["class"]}, \n instance: {instance}')
                    for j, dataloader in enumerate([nominal_dataloader, anominal_dataloader]):
                        projections = []

                        # if j == 2:
                        #     for _ in range(100):
                        #         noise_image = 2*torch.rand((config['batch_size'], 3, config['height'], config['width'])).cuda() - 1
                        #         z = encoder(noise_image)
                        #         projection = z @ random_unit_vector
                        #         projections.append(projection)
                        # else:
                        for (x, l) in dataloader:
                            x = x.cuda()
                            z = encoder(x)
                            projection = z @ random_unit_vector
                            projections.append(projection)

                        projections = torch.cat(projections)

                        ax[axis_index, i].hist(projections.cpu().numpy(), bins=100, density=True, histtype='step', color=('g' if j == 0 else ('r' if j == 1 else 'b')), label='nomaly' if j == 0 else 'anamoly')
                        if i == 1 and j == 0:
                            ax[axis_index, 2].hist(projections.cpu().numpy(), bins=100, density=True, histtype='step', color='g', label='nomaly')

                projections = []
                for (x, l) in augmented_nominal_dataloader:
                    x = x.cuda()
                    z = trained_encoder(x)
                    projection = z @ random_unit_vector
                    projections.append(projection)

                projections = torch.cat(projections)
                ax[axis_index, 2].hist(projections.cpu().numpy(), bins=100, density=True, histtype='step', color='b', label='augmented nominal')
                ax[axis_index, 2].title.set_text(f'augmented nominal, nominal')
                ax[axis_index, 3].plot(training_result.condition_no_list)
                ax[axis_index, 3].title.set_text(f'area under condition no: {area_under_curve(training_result.condition_no_list)}')
                roc = torch.stack(training_result.roc_list).cpu().numpy()
                ax[axis_index, 4].plot(roc[:, 0], label='p(z|u,c)')
                ax[axis_index, 4].plot(roc[:, 1], label='p(z|0,1)')
                ax[axis_index, 4].plot(roc[:, 2], label='||z||')
                ax[axis_index, 4].title.set_text(f'rocs')


            with open(analysis_csv_name, 'a', encoding='UTF8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([_class] + before_training_roc + after_training_roc)
                if axis_index == len(classes) - 1:
                    before_training_average_roc = before_training_sum_roc / len(classes)
                    after_training_average_roc = after_training_sum_roc / len(classes)
                    before_training_average_roc = list(np.round(before_training_average_roc.cpu().numpy(), 2))
                    after_training_average_roc = list(np.round(after_training_average_roc.cpu().numpy(), 2))
                    writer.writerow(['average'] + before_training_average_roc + after_training_average_roc)


    # plt.show()
    hlist, llist = [], []
    handles, labels = ax[0, 1].get_legend_handles_labels()
    hlist.append(handles)
    llist.append(labels)
    handles, labels = ax[0, 4].get_legend_handles_labels()
    hlist.append(handles)
    llist.append(labels)
    fig.legend(handles, labels, loc='upper center')
    plt.savefig(f'{folder_name}plot.png')


def generate_filter_plots():
    folder_name = 'model_cifar_20/run_4/'

    classes = list(range(10))
    classes.remove(3)
    plt.rcParams['font.size'] = '12'

    for axis_index, _class in enumerate(classes):
        fig, ax = plt.subplots(5, 5, figsize=(5, 5))
        config = {'height': 64, 'width': 64, 'batch_size': 64, 'n_critic': 5, 'clip': 1e-2, 'learning_rate': 5e-5,
                  'encoder_iters': (int)(10000), 'z_dim': 20, 'dataset': 'cifar', 'var_scale': 1,
                  'timestamp': (int)(time())}
        config['class'] = _class

        without_training_encoder = Encoder(config).cuda()
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
        without_training_encoder.apply(weights_init)

        with open(folder_name + f'cifar_{config["class"]}', 'rb') as file:
            training_result = pickle.load(file)

        trained_encoder = Encoder(config).cuda()
        trained_encoder.load_state_dict(training_result.min_condition_no_model)

        layer_1_weights = list(trained_encoder.parameters())[6]
        layer_1_kernels = layer_1_weights.reshape(512*256, 4, 4)
        random_indexes = (torch.rand((25))*layer_1_kernels.shape[0]).to(torch.long)
        random_kernels = layer_1_kernels[random_indexes, :, :].reshape(5, 5, 4, 4)

        fig.suptitle(f'class: {_class}')

        for i in range(5):
            for j in range(5):
                ax[i, j].imshow(random_kernels[i, j].detach().cpu().numpy(), cmap='gray')
                ax[i, j].axis('off')

        plt.savefig(f'{folder_name}plot.png')

if __name__ == '__main__':
    generate_plots_and_csv()

    # generate_filter_plots()