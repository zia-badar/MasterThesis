import sys

import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from SVDD.dataset import FilteredDataset
from SVDD.loss import Loss
from SVDD.model import Model


def train(dataloader, model, loss, c):
    model.train()
    # searching phase
    epochs = 150
    optimizer = Adam(list(model.parameters()), lr=1e-4, weight_decay=1e-6)
    for _ in tqdm(range(epochs)):
        for batch in dataloader:
            optimizer.zero_grad()
            l = loss(model, batch, c)
            l.backward()
            optimizer.step()

    # fine-tuning phase
    epochs = 100
    optimizer = Adam(list(model.parameters()), lr=1e-5, weight_decay=1e-6)
    for _ in tqdm(range(epochs)):
        for batch in dataloader:
            optimizer.zero_grad()
            l = loss(model, batch, c)
            l.backward()
            optimizer.step()

def test(dataloader, model, c, testing_class):
    scores = []
    labels = []
    with torch.no_grad():
        model.eval()
        for x, l in tqdm(dataloader):
            f_x = model(x)
            l = (l != testing_class) * 1
            scores.append(((f_x - c)**2).sum(dim=1))
            labels.append(l)

    scores = torch.cat(scores).detach().numpy()
    labels = torch.cat(labels).detach().numpy()

    result = roc_auc_score(labels, scores)
    print(testing_class, result)

    pass

def compute_c(dataloader, model):
    output = []
    for x, _ in dataloader:
        output.append(model(x))

    return (torch.cat(output).mean(dim=0)).detach()

if __name__ == '__main__':
    batch_size = 200
    for i in range(10):
        train_dataset = FilteredDataset(MNIST(root='', train=True, download=True), include_class_label=i, size=6000)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = Model(model_type='mnist')
        SVDDLoss = Loss()
        c = compute_c(train_dataloader, model)

        train(train_dataloader, model, SVDDLoss, c)

        test_dataset = FilteredDataset(MNIST(root='', train=False, download=True), exclude_class_label=11, size=10000)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test(test_dataloader, model, c, testing_class=i)