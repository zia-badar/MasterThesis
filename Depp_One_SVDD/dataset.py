import random

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class FilteredDataset(Dataset):
    
    MNIST_MIN_MAX = [[-0.8826568126678467, 9.001545906066895], [-0.666146457195282, 20.108057022094727], [-0.7820455431938171, 11.665099143981934], [-0.764577329158783, 12.895051956176758], [-0.7253923416137695, 12.683235168457031], [-0.7698502540588379, 13.103279113769531], [-0.7784181237220764, 10.457836151123047], [-0.7129780650138855, 12.057777404785156], [-0.828040361404419, 10.581538200378418], [-0.7369959950447083, 10.697040557861328]]
    CIFAR10_MIN_MAX = [[-28.94080924987793, 13.802960395812988], [-6.681769371032715, 9.158066749572754], [-34.92462158203125, 14.419297218322754], [-10.59916877746582, 11.093188285827637], [-11.945022583007812, 10.628044128417969], [-9.691973686218262, 8.94832706451416], [-9.174939155578613, 13.847018241882324], [-6.876684188842773, 12.28237247467041], [-15.603508949279785, 15.246490478515625], [-6.132884502410889, 8.046097755432129]]

    def __init__(self, dataset:Dataset, min_max_norm_class, include_class_label=None, exclude_class_label=None, size=None):
        assert include_class_label == None or exclude_class_label == None, 'one of include or exclude class label should be none'
        self.min_max_norm_class = min_max_norm_class

        self.dataset = dataset
        self.filtered_indexes = []
        self.to_tensor = ToTensor()
        for i, (x, l) in enumerate(self.dataset):
            if include_class_label != None and l == include_class_label:
                self.filtered_indexes.append(i)
            elif exclude_class_label != None and l != exclude_class_label:
                self.filtered_indexes.append(i)

            if size != None and len(self.filtered_indexes) > size:
                break

        random.shuffle(self.filtered_indexes)
        self.filtered_indexes = self.filtered_indexes[:size] if size != None else self.filtered_indexes

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]
        x = self.to_tensor(x)

        # l1 global contrastive normalization https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
        x_ = torch.mean(x)
        x = (x - x_) / torch.mean(torch.abs(x - x_))

        if type(self.dataset) == torchvision.datasets.mnist.MNIST:
            min = FilteredDataset.MNIST_MIN_MAX[self.min_max_norm_class][0]
            max = FilteredDataset.MNIST_MIN_MAX[self.min_max_norm_class][1]
            x = (x - min) / (max - min)
        elif type(self.dataset) == torchvision.datasets.cifar.CIFAR10:
            min = FilteredDataset.CIFAR10_MIN_MAX[self.min_max_norm_class][0]
            max = FilteredDataset.CIFAR10_MIN_MAX[self.min_max_norm_class][1]
            x = (x - min) / (max - min)

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)
