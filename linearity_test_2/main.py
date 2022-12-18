import torch.nn
from torch import softmax, sigmoid
from torch.distributions import MultivariateNormal
from torch.linalg import eig
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from linearity_test_2.models import Discriminator, Encoder
from linearity_test_2.datasets import ProjectedDataset


def train_encoder(config):
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])

    train_dataset = ProjectedDataset(train=True)
    validation_dataset = ProjectedDataset(train=True)

    f = Discriminator(config).cuda()
    e = Encoder(config).cuda()

    def _next(iter):
        try:
            batch = next(iter)
            return False, batch
        except StopIteration:
            return True, None

    discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20))
    encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20))
    optim_f = SGD(f.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optim_e = SGD(e.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    normal_dist = MultivariateNormal(loc=torch.zeros(config['encoding_dim']), covariance_matrix=torch.eye(config['encoding_dim']))
    mean, condition_no = evaluate_encoder(e, train_dataset, validation_dataset, config)
    print(f'mean: {torch.norm(mean).item(): .4f}, condition_no: {condition_no.item(): .4f}')

    for encoder_iter in range(1, config['encoder_iters']+1):

        for _ in range(config['discriminator_n']):
            empty, batch = _next(discriminator_dataloader_iter)
            if empty:
                discriminator_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20))
                _, batch = _next(discriminator_dataloader_iter)

            x, _ = batch
            x = x.cuda()
            z = normal_dist.sample((x.shape[0], )).cuda()

            loss = -torch.mean(f(z) - f(e(x)))

            optim_f.zero_grad()
            loss.backward()
            optim_f.step()

            for parameter in f.parameters():
                parameter.data.clamp(-config['clip'], config['clip'])

        empty, batch = _next(encoder_dataloader_iter)
        if empty:
            encoder_dataloader_iter = iter(DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20))
            _, batch = _next(encoder_dataloader_iter)

        x, _ = batch
        x = x.cuda()

        loss = -torch.mean(f(e(x)))

        optim_e.zero_grad()
        loss.backward()
        optim_e.step()

        if encoder_iter % 100 == 0:
            mean, condition_no = evaluate_encoder(e, train_dataset, validation_dataset, config)
            print(f'mean: {torch.norm(mean).item(): .4f}, condition_no: {condition_no.item(): .4f}')
            torch.save(e.state_dict(), f'encoder_{config["class"]}')

def evaluate_encoder(encoder, train_dataset, validation_dataset, config):
    encoder.eval()

    with torch.no_grad():
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
        encodings = []
        for x, _ in train_dataloader:
            x = x.cuda()
            encodings.append(encoder(x))

        encodings = torch.cat(encodings)
        mean = torch.mean(encodings, dim=0)
        # print(f'std: {torch.std(encodings, dim=0)}')
        cov = torch.cov(encodings.t(), correction=0)
        eig_val, eig_vec = eig(cov)
        condition_no = torch.max(eig_val.real) / torch.min(eig_val.real)

    encoder.train()

    return mean, condition_no

if __name__ == '__main__':
    config = {'batch_size': 64, 'epochs': 200, 'data_dim': 64, 'encoding_dim': 8, 'encoder_iters': 1000000, 'discriminator_n': 4, 'lr': 1e-3, 'weight_decay': 1e-5, 'clip': 1e-2}

    config['class'] = 0
    # train_classifier(config)
    # train_binary_classifier(config)
    train_encoder(config)

    # dataset = CIFAR10(root='../', train=True, transform=ToTensor())
    # validation_dataset = Subset(dataset, range((int)(0.7 * len(dataset)), len(dataset)))
    # outlier_classes = list(range(10))
    # outlier_classes.remove(config['class'])
    # validation_dataset = OneClassDataset(validation_dataset, one_class_labels=[config['class']], zero_class_labels=outlier_classes)
    # model = classifier(1)
    # model.load_state_dict(torch.load('cifar_binary_0_classifier'))
    # print(f'accuracy: {evaluate_one_class_classifier(model, validation_dataset, config) : .4f}')