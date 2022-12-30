import torch.distributions
from matplotlib import pyplot
from matplotlib.pyplot import plot
from torch import nn, tanh
from torch.distributions import MultivariateNormal, Uniform
from torch.optim import Adam


class ScaledTanh(nn.Module):

    def __init__(self):
        super(ScaledTanh, self).__init__()

        # self.scale = 10
        self.scale = 100

    def forward(self, x):
        y = torch.tanh(x * self.scale)
        # y = y + (x > 0.2) * 0.01 * (torch.log(torch.exp(x)) - 0.2)
        # y = y + (x < 0.2) * 0.01 * (torch.log(torch.exp(x)) + 0.2)
        y = y + (x > 0.06) * 0.02 * (torch.log(torch.exp(x)) - 0.06)
        y = y + (x < 0.06) * 0.02 * (torch.log(torch.exp(x)) + 0.06)
        return y

class AbsActivation(nn.Module):

    def __init__(self):
        super(AbsActivation, self).__init__()

        self.base = 0.0
        self.slope = 0.1

    def forward(self, x):
        ret = self.base + torch.abs(x) * self.slope
        return ret

def train(config):

    model = nn.Sequential(
        # nn.Linear(config['encoding_dim'], 2*config['batch_size']*config['encoding_dim']),
        # nn.Linear(config['encoding_dim'], 2*2*config['encoding_dim']),
        nn.Linear(config['encoding_dim'], 2),
        AbsActivation(),
        # ScaledTanh(),
        # nn.Linear(2*2*config['encoding_dim'], 1)
    )

    normal = MultivariateNormal(torch.zeros(config['encoding_dim']), torch.eye(config['encoding_dim']))
    # normal = Uniform(0, 1)
    # data = normal.sample((config['batch_size'], ))
    # data = normal.sample((1, ))
    data = torch.tensor([0.1, 0.9]).unsqueeze(-1)

    optim = Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)
    for _ in range(config['iterations']):
        random = normal.sample((config['batch_size'],)).unsqueeze(-1)
        f_data = torch.mean(torch.mean(model(data), dim=-1))
        f_random = torch.mean(torch.mean(model(random), dim=-1))
        loss = -(f_random - f_data)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f'loss: {loss.item(): .4f}')

    return model, data

def analyse(model, data):

    x = torch.arange(-1, 1, 0.01).unsqueeze(-1)
    # y = AbsActivation()(x)

    with torch.no_grad():
        y = torch.sum(model(x), dim=-1)
    x = x.numpy()
    y = y.numpy()

    pyplot.scatter(x, y, c='g')
    print(f'data: {data}')

    pyplot.show()


if __name__ == '__main__':
    config = {'encoding_dim': 1, 'batch_size': 10, 'iterations': 1000}

    model, data = train(config)

    analyse(model, data)