import torch
from torch.utils.data import Dataset

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

    def __getitem__(self, item):
        x, l = self.dataset[self.filtered_indexes[item]]

        if self.transform != None:
            x = self.transform(x)
        l = 1 if l in self.one_class_labels else 0

        return x, l

    def __len__(self):
        return len(self.filtered_indexes)

class EmbeddedDataset(Dataset):
    def __init__(self, dataset, classifier):
        super(EmbeddedDataset, self).__init__()

        self.dataset = dataset
        self.classifier = classifier
        self.classifier.eval()

        es = []
        with torch.no_grad():
            for x, _ in dataset:
                e, _ = self.classifier(torch.unsqueeze(x, dim=0))
                e = torch.squeeze(e)
                es.append(e)

        es = torch.stack(es)
        mean = torch.mean(es, dim=0)
        std = torch.std(es, dim=0)
        self.es = (es - mean)/std

    def __getitem__(self, item):
        x, l = self.dataset[item]

        # with torch.no_grad():
        #     e, _ = self.classifier(torch.unsqueeze(x, dim=0))
        #     e = torch.squeeze(e)

        return self.es[item], l

        # return e, l

    def __len__(self):
        return len(self.dataset)