import multiprocessing
from copy import copy

import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from fcdd.dataset import FilteredDataset, MergedDataset
from fcdd.loss import Loss
from fcdd.model import Model


def train(dataloader, model, loss):

    model.train()
    for epochs, lr in zip([400, 100, 100], [1e-3, 1e-4, 1e-5]):
        optimizer = Adam(list(model.parameters()), lr=lr, weight_decay=1e-6)
        tqdm_bar = tqdm(range(epochs), leave=False)
        for _ in tqdm_bar:
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                l = loss(model, batch)
                l.backward()
                optimizer.step()
                total_loss += l.item()

            tqdm_bar.set_description(f'epochs, loss: {total_loss : .4f}')

def test(dataloader, model):

    scores = []
    labels = []
    with torch.no_grad():
        model.eval()
        for x, l in tqdm(dataloader, desc='testing...', leave=False):
            x = x.cuda()
            A_X = torch.sqrt(model(x) ** 2 + 1) - 1
            score = torch.sum(A_X, dim=(1, 2, 3))
            scores.append(score)
            labels.append(l)

    scores = torch.cat(scores).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    result = roc_auc_score(labels, scores)
    return result

batch_size = 200
results_file = 'output_results.log'
cifar_labels = CIFAR10(root='../', train=True, download=True).classes
def process_class(_class):
    print(f'processing class: {_class}')
    outlier_exposure_classes = list(range(0, 101))
    inlier_dataset = FilteredDataset(CIFAR10(root='../', train=True, download=True), [_class], [], _class)
    outlier_exposure_dataset = FilteredDataset(CIFAR100(root='../', train=True, download=True),
                                               outlier_exposure_classes, outlier_exposure_classes, _class)
    train_dataset = MergedDataset([inlier_dataset, outlier_exposure_dataset])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    model = Model('cifar').cuda()
    loss = Loss()
    train(train_dataloader, model, loss)

    all_classes = list(range(0, 10))
    outlier_classes = copy(all_classes)
    outlier_classes.remove(_class)
    test_dataset = FilteredDataset(CIFAR10(root='../', train=False, download=True), all_classes, outlier_classes, _class)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
    roc_auc = test(test_dataloader, model)

    with open(results_file, 'a') as file:
        file.write(f'class: {_class} : {cifar_labels[_class]}, roc_auc: {roc_auc : .2f}\n')

    print(f'class: {_class} : {cifar_labels[_class]}, roc_auc: {roc_auc: .2f}\n')

# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc

if __name__ == '__main__':
    with NoDaemonProcessPool(processes=10) as pool:
        for _ in pool.imap_unordered(process_class, range(10)):
            pass