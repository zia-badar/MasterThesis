import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset

from linearity_test_1.models import classifier


class ProjectedDataset(Dataset):

    def __init__(self, train, distribution, projection, translation):
        super(ProjectedDataset, self).__init__()

        self.distribution = distribution
        self.projection = projection
        self.translation = translation

        if train:
            self.dataset = self.distribution.sample(sample_shape=(5000,)) @ self.projection + self.translation
        else:
            self.dataset = self.distribution.sample(sample_shape=(1500,)) @ self.projection + self.translation

    def __getitem__(self, item):

        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
