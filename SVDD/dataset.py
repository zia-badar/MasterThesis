import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class FilteredDataset(Dataset):
    
    MNIST_MIN_MAX = [[-0.8826568126678467, 9.001545906066895], [-0.666146457195282, 20.108057022094727], [-0.7820455431938171, 11.665099143981934], [-0.764577329158783, 12.895051956176758], [-0.7253923416137695, 12.683235168457031], [-0.7698502540588379, 13.103279113769531], [-0.7784181237220764, 10.457836151123047], [-0.7129780650138855, 12.057777404785156], [-0.828040361404419, 10.581538200378418], [-0.7369959950447083, 10.697040557861328]]

    def __init__(self, dataset:Dataset, include_class_label=None, exclude_class_label=None, size=None):
        assert include_class_label == None or exclude_class_label == None, 'one of include or exclude class label should be none'

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

        self.filtered_indexes = self.filtered_indexes[:size] if size != None else self.filtered_indexes

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]
        x = self.to_tensor(x)

        if type(self.dataset) == torchvision.datasets.mnist.MNIST:
            # l1 global contrastive normalization https://cedar.buffalo.edu/~srihari/CSE676/12.2%20Computer%20Vision.pdf
            x_ = torch.mean(x)
            x = (x - x_) / torch.mean(torch.abs(x - x_))

            min = FilteredDataset.MNIST_MIN_MAX[l][0]
            max = FilteredDataset.MNIST_MIN_MAX[l][1]
            x = (x - min) / (max - min)

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)
