import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MergedDataset(Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        item += 1
        for dataset in self.datasets:
            if item <= len(dataset):
                return dataset[item-1]
            item -= len(dataset)

    def __len__(self):
        l = 0
        for dataset in self.datasets:
            l += len(dataset)
        return l

class FilteredDataset(Dataset):
    CIFAR10_MEAN = [[0.5256556, 0.56033057, 0.58890575], [0.47118282, 0.45452955, 0.4471987], [0.48925033, 0.49147782, 0.42404503], [0.4954817, 0.45641166, 0.41553804], [0.47159043, 0.46520537, 0.37820706], [0.49992543, 0.4646367, 0.4165468], [0.4700562, 0.43839356, 0.3452189], [0.5019589, 0.47986475, 0.41688657], [0.4902258, 0.5253955, 0.5546858], [0.49866694, 0.4853417, 0.47807592]]
    CIFAR10_STD = [[0.2502202, 0.24083486, 0.2659735], [0.26806358, 0.26582742, 0.27494594], [0.2270548, 0.2209446, 0.24337928], [0.25684315, 0.25227082, 0.25799376], [0.21732737, 0.20652702, 0.21182336], [0.25042534, 0.24374878, 0.24894638], [0.22888342, 0.21856172, 0.22041996], [0.24304901, 0.24397305, 0.25171563], [0.24962473, 0.24068885, 0.25149763], [0.26805255, 0.269108, 0.28101656]]

    def __init__(self, dataset: Dataset, classes_to_consider, outlier_classes, normalization_class):
        self.dataset = dataset

        self.filtered_indexes = []
        for i, (x, l) in enumerate(self.dataset):
            if l in classes_to_consider:
                self.filtered_indexes.append(i)

        # from original implementation
        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
            transforms.Normalize(FilteredDataset.CIFAR10_MEAN[normalization_class], FilteredDataset.CIFAR10_STD[normalization_class])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FilteredDataset.CIFAR10_MEAN[normalization_class], FilteredDataset.CIFAR10_STD[normalization_class])
        ])

        self.target_transform = transforms.Compose([transforms.Lambda(lambda x: (int)(x in outlier_classes))])

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]

        x = self.train_transform(x) if self.dataset.train else self.test_transform(x)
        l = self.target_transform(l)

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)
