import torch
from torch.utils.data import Dataset


class OneClassDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, class_label, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.filtered_indexes = []

        for i, (x, l) in enumerate(self.dataset):
            if l == class_label:
                self.filtered_indexes.append(i)

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]

        if self.transform != None:
            x = self.transform(x)

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)
