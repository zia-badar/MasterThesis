import torch
from sklearn.metrics import roc_auc_score
from torch.nn.functional import normalize
from torch.optim import Adam
from tqdm import tqdm

from CSI.loss import Loss
from CSI.utils import kmean_coreset


to do
def test(train_dataloader, test_dataloader, model):
    nonlocal config

    with torch.no_grad():
        model.eval()
        z_xm = kmean_coreset(train_dataloader, model)           # |shift| x |m| x |encoded_dim|
        lambda_s = z_xm.shape[1] / torch.sum(torch.norm(z_xm, dim=2), dim=1)    # |shift|

        scores = []
        labels = []
        for (x, l) in test_dataloader:
            x = x.cuda()

            x = x.reshape(config.batch_size, config.shift_count, config.align_count, config.channel, config.height, config.width)[:, :, :1, :, :].squeeze()
            z = model(x.reshape(-1, config.height, config.width)).rehape(config.batch_size, config.shift_count, config.encoded_dims)    # |batch_size| x |shift| x |encoded_dim|
            s_con = torch.max(torch.squeeze(normalize(z_xm, dim=-1) @ torch.unsqueeze(normalize(z, dim=-1), -1), dim=-1), dim=2) * torch.norm(z, dim=2)
            s_con_si = s_con @ lambda_s

            scores.append(s_con_si)
            labels.append(l)

        scores = torch.stack(scores)
        labels = torch.stack(labels)

    return roc_auc_score(labels, scores)

def train(dataloader, model, epochs):
    nonlocal config

    model.train()
    optim = Adam(list(model.getParameters()), lr=1e-3, weight_decay=1e-5)
    loss = Loss(model)

    epoch_progress_bar = tqdm(range(epochs))
    for e in epoch_progress_bar:
        total_loss = 0
        for batch in tqdm(dataloader, leave=False, desc='training'):
            optim.zero_grad()
            l = loss(batch)
            l.backward()
            optim.step()
            total_loss += l.item()

        epoch_progress_bar.set_description(f'epoch: {e}, loss: {total_loss/(len(dataloader.dataset)*config.shift_count) : .4f}')

    return model

if __name__ == '__main__':
