import torch
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal


def get_variance(train_dataloader, e):
    with torch.no_grad():
        e.eval()
        zs = []
        for _x, _l in train_dataloader:
            _x = _x.cuda()
            zs.append(e(_x))

        zs = torch.cat(zs)
        # mean = torch.mean(zs, dim=0)
        # var = torch.cov(zs.T, correction=0)
        # dist = MultivariateNormal(loc=mean, covariance_matrix=var)

        var_samples = []
        for i in range(100):
            random_unit = torch.rand_like(zs[0])
            random_unit /= torch.norm(random_unit)

            projections = zs @ random_unit
            var_samples.append(torch.var(projections).item())

        # print(torch.tensor(var_samples).mean())
        # plt.hist(projections.detach().cpu().numpy())
        # plt.show()
        e.train()

    return torch.tensor(var_samples).mean().item()
