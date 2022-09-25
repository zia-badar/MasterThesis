import torch


class Loss():
    def __init__(self):
        pass

    def __call__(self, model, batch):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        A_X = torch.sqrt(model(x)**2 + 1) -1
        u, v = list(A_X.shape[2:])
        A_X_norm = torch.sum(A_X, dim=(1, 2, 3))

        loss = (1-y)*(1/(u*v))*A_X_norm - y*torch.log(1 - torch.exp(-(1/(u*v))*A_X_norm))
        loss = torch.mean(loss)
        return loss