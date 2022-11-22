import multiprocessing
import os
from time import time

import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms, ToTensor, Resize, Normalize
from tqdm import tqdm

import sys

sys.path.append('/home/zia/Desktop/MasterThesis/')

from thesis.dataset import OneClassDataset, GlobalContrastiveNormalizationTransform, MinMaxNormalizationTransform
from thesis.models import Encoder, Decoder


# min max value for each class after applying global contrastive normalization
CIFAR10_MIN_MAX = [[-28.94080924987793, 13.802960395812988], [-6.681769371032715, 9.158066749572754],
                   [-34.92462158203125, 14.419297218322754], [-10.59916877746582, 11.093188285827637],
                   [-11.945022583007812, 10.628044128417969], [-9.691973686218262, 8.94832706451416],
                   [-9.174939155578613, 13.847018241882324], [-6.876684188842773, 12.28237247467041],
                   [-15.603508949279785, 15.246490478515625], [-6.132884502410889, 8.046097755432129]]


def train(config):
    inlier = [config['class']]
    outlier = list(range(10))
    outlier.remove(config['class'])

    dataset = CIFAR10(root='../', train=True, download=True)
    normlization_transforms = transforms.Compose([ToTensor(), Resize((config['height'], config['width'])), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    inlier_dataset = OneClassDataset(dataset, zero_class_labels=inlier, transform=normlization_transforms)
    outlier_dataset = OneClassDataset(dataset, one_class_labels=outlier, transform=normlization_transforms)

    train_inlier_dataset = Subset(inlier_dataset, range(0, (int)(.7 * len(inlier_dataset))))
    validation_inlier_dataset = Subset(inlier_dataset, range((int)(.7 * len(inlier_dataset)), len(inlier_dataset)))
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])

    train_dataset = train_inlier_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20, pin_memory=True)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    encoder = Encoder(config).cuda()
    decoder = Decoder(config).cuda()
    encoder.train()
    decoder.train()
    encoder.apply(weights_init)
    decoder.apply(weights_init)

    optim = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config['learning_rate'], weight_decay=1e-6)
    for epoch in tqdm(range(config['autoencoder_epochs'])):
        total_loss = 0
        for x, _ in train_dataloader:
            optim.zero_grad()
            x = x.cuda(non_blocking=True)
            encoded = encoder(x.cuda())
            decoding = decoder(encoded)
            diff = torch.sum(((x - decoding) ** 2), dim=[1, 2, 3])                                          # sum diff across channel, height, width dimensions
            loss = torch.mean(diff)
            loss.backward()
            optim.step()
            total_loss += loss.item()

    roc_auc = evaluate(validation_dataset, encoder, decoder, config)
    print(f'epoch: {epoch}, class: {config["class"]}, encoder training, loss: {total_loss/len(train_dataloader): .2f}, {roc_auc}')
    torch.save(encoder.state_dict(), result_directory + f'/encoder_class_{config["class"]}')

def evaluate(validation_dataset, encoder, decoder, config):

    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        scores = []
        targets = []
        for x, l in validation_dataloader:
            x = x.cuda()
            encoded = encoder(x)
            decoded = decoder(encoded)
            score = torch.sum((x - decoded)**2, dim=(1, 2, 3))
            scores.append(score)
            targets.append(l)

        scores = torch.cat(scores).cpu().numpy()
        targets = torch.cat(targets).cpu().numpy()

        roc_auc = roc_auc_score(targets, scores)

        encoder.train()
        decoder.train()

    return roc_auc

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
    result_directory = f'model_autoencoder/run_{(int)(time())}'
    os.mkdir(result_directory)
    config = {'height': 64, 'width': 64, 'batch_size': 200, 'learning_rate': 5e-5, 'autoencoder_epochs': (int)(250), 'z_dim': 20, 'dataset': 'cifar', 'result_directory': result_directory}

    for j in range(0, 10, 2):
        with NoDaemonProcessPool(processes=10) as pool:
            configs = []
            for i in range(j, j+2):
                _config = config.copy()
                _config['class'] = i
                configs.append(_config)

            for _ in pool.imap_unordered(train, configs):
                pass