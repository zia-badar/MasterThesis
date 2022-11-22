from random import randrange

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import rotate


class OneClassDataset(Dataset):
    def __init__(self, dataset: Dataset, one_class_labels=[], zero_class_labels=[], transform=None):
        self.dataset = dataset
        self.one_class_labels = one_class_labels
        self.transform = transform
        self.filtered_indexes = []

        valid_labels = one_class_labels + zero_class_labels
        for i, (x, l) in enumerate(self.dataset):
            if l in valid_labels:
                self.filtered_indexes.append(i)

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]

        x = self.transform(x)
        l = 1 if l in self.one_class_labels else 0

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)

class Random90RotationTransform:
    def __init__(self):
        self.angles = [90, 180, 270]

    def __call__(self, x):
        x = rotate(x, angle=self.angles[randrange(len(self.angles))])
        return x

class RandomPermutationTransform:
    def __init__(self):
        self.no_splits = torch.tensor([4, 4])
        self.image_size = torch.tensor([64, 64])
        self.split_size = (self.image_size / self.no_splits).int()

    def __call__(self, x):
        x = x.unfold(1, self.split_size[0], self.split_size[0]).unfold(2, self.split_size[1], self.split_size[1]).reshape( [-1, self.no_splits.prod(), self.split_size[0], self.split_size[1]])[:, torch.randperm(self.no_splits.prod())]
        x = x.unfold(1, self.no_splits[0], self.no_splits[0]).permute([0, 4, 2, 1, 3]).reshape(-1, self.image_size[0], self.image_size[1])
        return x


class GlobalContrastiveNormalizationTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # l1 global contrastive normalization https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
        x_ = torch.mean(x)
        x = (x - x_) / torch.mean(torch.abs(x - x_))
        return x

class MinMaxNormalizationTransform:
    def __init__(self, min_max):
        self.min = min_max[0]
        self.max = min_max[1]

    def __call__(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x

class RandomBoxDataset(Dataset):
    def __init__(self, size=(32, 32)):
        self.resize = Resize(size=size)
        self.dist = torch.distributions.normal.Normal(torch.tensor([0.]), torch.tensor([1.]))
        self.scalling = 7
        self.image_center = torch.tensor([size[0]/2, size[0]/2])
        self.shape = torch.tensor(list(size))
        self.spot_size = torch.tensor([5, 5])
        self.spot_index = (torch.ones(size=(self.spot_size[0], self.spot_size[1])) == 1).nonzero()
        self.spot_index = self.spot_index - torch.floor(self.spot_size/2)

    def __getitem__(self, item):
        x = torch.zeros(size=(self.shape[0], self.shape[1]))
        rand_ind = self.image_center + self.dist.sample((2,))[:, 0] * self.scalling
        rand_ind = (rand_ind + self.spot_index).type(torch.int32)
        mask = torch.logical_and(torch.logical_and(rand_ind[:, 0] >= 0, rand_ind[:, 0] < 32), torch.logical_and(rand_ind[:, 1] >= 0, rand_ind[:, 1] < 32))
        rand_ind = rand_ind[mask, :].type(torch.long)
        x[rand_ind[:, 0], rand_ind[:, 1]] = 1

        x = x*2 - 1         # scalling

        return x.unsqueeze(0), 0

    def __len__(self):
        return 32 * 1000
