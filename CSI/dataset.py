import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, dataset: Dataset, align_transforms, shift_transforms):
        self.dataset = dataset
        self.align_transforms = align_transforms
        self.shift_transforms = shift_transforms

    def __getitem__(self, item):
        x, l = self.dataset[item]

        xs = []
        for shift_transform in self.shift_transforms:
            shifted_x = shift_transform(x)
            xt = []
            xt.append(shifted_x)
            for align_transform in self.align_transforms:
                xt.append(align_transform(shifted_x))

            xs.append(torch.unsqueeze(torch.stack(xt), 0))

        xs = torch.stack(xs)

        return xs, l        #   shift_count x (align_count+1) x channel x height x width

    def __len__(self):
        return len(self.dataset)