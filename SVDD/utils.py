import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from SVDD.dataset import FilteredDataset


def min_max():
    min_max = []
    for i in range(10):
        train_dataset = FilteredDataset(MNIST(root='', train=True, download=True), include_class_label=i)
        train_dataloader = DataLoader(train_dataset, batch_size=200, shuffle=True)

        overall_min = sys.maxsize
        overall_max = -sys.maxsize
        for x, _ in train_dataloader:
            new_min = torch.min(x)
            if new_min < overall_min:
                overall_min = new_min
            new_max = torch.max(x)
            if new_max > overall_max:
                overall_max = new_max

        min_max.append([overall_min.item(), overall_max.item()])

    return min_max