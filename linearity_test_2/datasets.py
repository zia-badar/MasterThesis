import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset

from linearity_test_1.models import classifier


class ProjectedDataset(Dataset):

    def __init__(self, train):
        super(ProjectedDataset, self).__init__()

        low_dim = 8
        high_dim = 64
        distribution = MultivariateNormal(loc=torch.zeros(low_dim), covariance_matrix=torch.eye(low_dim))
        projection = torch.rand(size=(low_dim, high_dim))
        translation = 2*torch.rand(size=(high_dim,))

        if train:
            self.dataset = distribution.sample(sample_shape=(5000,)) @ projection + translation
        else:
            self.dataset = distribution.sample(sample_shape=(1500,)) @ projection + translation

    def __getitem__(self, item):

        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
