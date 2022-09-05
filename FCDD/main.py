import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from FCDD.dataset import DatasetWrapper
from FCDD.loss import Loss
from FCDD.model import Model


def train(dataloader, model, loss):

    model.train()
    for epochs, lr in zip([400, 100, 100], [1e-3, 1e-4, 1e-5]):
        total_loss = 0
        optimizer = Adam(list(model.parameters()), lr=lr, weight_decay=1e-6)
        tqdm_bar = tqdm(range(epochs))
        for _ in tqdm_bar:
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
        for x, l in dataloader:
            x = x.cuda()
            A_X = torch.sqrt(model(x) ** 2 + 1) - 1
            score = torch.sum(A_X, dim=(1, 2, 3))
            scores.append(score)
            labels.append(l)

    scores = torch.cat(scores).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()

    result = roc_auc_score(labels, scores)
    return result

if __name__ == '__main__':

    batch_size = 200
    for _class in range(10):
        train_dataset = DatasetWrapper(CIFAR10(root='../', train=True, download=True), normal_class=_class)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = Model('cifar').cuda()
        loss = Loss()
        train(train_dataloader, model, loss)

        test_dataset = DatasetWrapper(CIFAR10(root='../', train=False, download=True), normal_class=_class)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=20)
        roc_auc = test(test_dataloader, model)
        print(f'class: {_class}, roc_auc: {roc_auc : .2f}')