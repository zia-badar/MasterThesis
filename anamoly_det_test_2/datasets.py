import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor

from linearity_test_1.models import classifier


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

        self.toTensor = ToTensor()

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]

        # x = self.transform(x)
        x = self.toTensor(x)
        l = 1 if l in self.one_class_labels else 0

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)

class EmbeddedDataset(Dataset):

    def __init__(self, dataset, model):
        with torch.no_grad():
            self.x = []
            self.l = []
            dataloader = DataLoader(dataset, shuffle=False, batch_size=128)
            for x, l in dataloader:
                f, _ = model(x.cuda())
                self.x.append(f)
                self.l.append(l)

            self.x = torch.cat(self.x)
            self.l = torch.cat(self.l)

        self.x = self.x.cpu()
        self.l = self.l.cpu()

    def __getitem__(self, item):
        return self.x[item], self.l[item]

    def __len__(self):
        return self.x.shape[0]

class AugmentedDataset(Dataset):
    def __init__(self, dataset, pair=True):
        self.dataset = dataset
        # self.augmentation = Random90RotationTransform()
        self.pair = pair

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2)])

    def __getitem__(self, item):
        x, l = self.dataset[item]

        if self.pair:
            x_aug = self.augmentation(x)

        x = self.augmentation(x)

        if self.pair:
            return x, x_aug, l
        else:
            return x, l

    def __len__(self):
        return len(self.dataset)
