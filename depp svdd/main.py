import sys

import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm

from Depp_One_SVDD.dataset import FilteredDataset
from Depp_One_SVDD.loss import Loss
from Depp_One_SVDD.model import Encoder, Decoder

def train_encoder(dataloader, model_type):

    encoder = Encoder(model_type).cuda()
    decoder = Decoder(model_type).cuda()
    encoder.train()
    decoder.train()

    # searching and fine tuning training phases
    for lr, epochs in zip([1e-4, 1e-5], [250, 100]):
        optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-6)
        tqdm_bar = tqdm(range(epochs), leave=False)
        for _ in tqdm_bar:
            total_loss = 0
            for x, _ in dataloader:
                optimizer.zero_grad()
                encoding = encoder(x.cuda())
                encoding_with_channel = encoding.view(x.shape[0], (int)(encoder.encode_dim / (4*4)), 4, 4)       # encoding with channels
                decoding = decoder(encoding_with_channel)
                diff = torch.sum(((x.cuda() - decoding) ** 2), dim=[1, 2, 3])                                          # sum diff across channel, height, width dimensions
                loss = torch.mean(diff)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            tqdm_bar.set_description(desc=f'encoder training, loss: {total_loss/len(dataloader): .4f}')

    return encoder

def train(dataloader, model, loss, c):
    model.train()
    # searching phase and fine tuning phase
    for lr, epochs in zip([1e-4, 1e-5], [150, 100]):
        optimizer = Adam(list(model.parameters()), lr=lr, weight_decay=1e-6)
        tqdm_bar = tqdm(range(epochs), leave=False)
        for _ in tqdm_bar:
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                l = loss(model, batch, c)
                l.backward()
                optimizer.step()
                total_loss += l.item()
            tqdm_bar.set_description(desc=f'One Class Deep SVDD training, loss: {total_loss/len(dataloader): .4f}')

# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.9777&rep=rep1&type=pdf
def test(dataloader, model, c, testing_class):
    scores = []
    labels = []
    with torch.no_grad():
        model.eval()
        for x, l in tqdm(dataloader, leave=False, desc='testing'):
            f_x = model(x.cuda())
            l = ((l != testing_class) * 1).cuda()
            scores.append(((f_x - c)**2).sum(dim=1))
            labels.append(l)

    scores = torch.cat(scores).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    result = roc_auc_score(labels, scores)
    return result

def compute_c(dataloader, model, eps=0.1):
    output = []
    with torch.no_grad():
        model.eval()
        for x, _ in dataloader:
            output.append(model(x.cuda()))

    c = (torch.cat(output).mean(dim=0)).detach()
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

if __name__ == '__main__':
    batch_size = 200
    for _class in range(10):
        roc_auc_runs = []
        for _ in range(10):
            train_dataset = FilteredDataset(MNIST(root='../', train=True, download=True), min_max_norm_class=_class, include_class_label=_class, size=6000)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
            model = train_encoder(train_dataloader, model_type='mnist')
            model.use_in_autoencoder = False
            SVDDLoss = Loss()
            c = compute_c(train_dataloader, model).cuda()

            train(train_dataloader, model, SVDDLoss, c)

            test_dataset = FilteredDataset(MNIST(root='../', train=False, download=True), min_max_norm_class=_class, exclude_class_label=11, size=10000)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
            roc_auc = test(test_dataloader, model, c, testing_class=_class)
            roc_auc_runs.append(roc_auc)

        roc_auc_runs = torch.tensor(roc_auc_runs)
        mean = torch.mean(roc_auc_runs)
        std = torch.std(roc_auc_runs)

        print(f'class: {_class}, results: {mean.item()} +- {std.item()}')


    for _class in range(10):
        roc_auc_runs = []
        for _ in range(10):
            train_dataset = FilteredDataset(CIFAR10(root='../', train=True, download=True), min_max_norm_class=_class, include_class_label=_class, size=5000)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
            model = train_encoder(train_dataloader, model_type='cifar')
            model.use_in_autoencoder = False
            SVDDLoss = Loss()
            c = compute_c(train_dataloader, model).cuda()

            train(train_dataloader, model, SVDDLoss, c)

            test_dataset = FilteredDataset(CIFAR10(root='../', train=False, download=True), min_max_norm_class=_class, exclude_class_label=11, size=10000)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
            roc_auc = test(test_dataloader, model, c, testing_class=_class)
            roc_auc_runs.append(roc_auc)

        roc_auc_runs = torch.tensor(roc_auc_runs)
        mean = torch.mean(roc_auc_runs)
        std = torch.std(roc_auc_runs)

        print(f'class: {_class}, results: {mean.item()} +- {std.item()}')