import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

class ProjectedDataset(Dataset):

    def __init__(self, train, distribution, projection):
        super(ProjectedDataset, self).__init__()

        projection.eval()
        with torch.no_grad():
            if train:
                self.dataset = projection(distribution.sample(sample_shape=(5000,)))
            else:
                self.dataset = projection(distribution.sample(sample_shape=(2000,)))

    def __getitem__(self, item):

        return self.dataset[item], 0

    def __len__(self):
        return len(self.dataset)
