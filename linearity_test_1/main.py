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

from linearity_test_1.datasets import OneClassDataset, EmbeddedDataset
from linearity_test_1.models import classifier, Discriminator, Encoder


def train_classifier(config):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = CIFAR10(root='../', train=True, transform=transform)
    train_dataset = Subset(dataset, range(0, (int)(0.7 * len(dataset))))
    validation_dataset = Subset(dataset, range((int)(0.7 * len(dataset)), len(dataset)))

    model = classifier(10).cuda()
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
    loss = CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config['epochs'])

    for epoch in range(1, config['epochs']+1):
        for x, y in train_dataloader:
            x = x.cuda()
            y = y.cuda()
            _, pred = model(x)
            l = loss(pred, y)
            optim.zero_grad()
            l.backward()
            optim.step()

        scheduler.step()
        print(f'epoch: {epoch}, accuracy: {evaluate_classifier(model, validation_dataset, config): .4f}')
    torch.save(model.state_dict(), './cifar_10_classifier')

def train_binary_classifier(config):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = CIFAR10(root='../', train=True, transform=transform)
    train_dataset = Subset(dataset, range(0, (int)(0.7 * len(dataset))))
    validation_dataset = Subset(dataset, range((int)(0.7 * len(dataset)), len(dataset)))
    outlier_classes = list(range(10))
    outlier_classes.remove(config['class'])
    train_dataset = OneClassDataset(train_dataset, one_class_labels=[config['class']], zero_class_labels=outlier_classes)
    validation_dataset = OneClassDataset(validation_dataset, one_class_labels=[config['class']], zero_class_labels=outlier_classes)

    model = classifier(1).cuda()
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
    loss = BCEWithLogitsLoss()
    optim = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=config['epochs'])

    for epoch in range(1, 2+1):
        total_loss = 0
        for x, y in train_dataloader:
            x = x.cuda()
            y = torch.as_tensor(y, dtype=torch.float).cuda()
            _, pred = model(x)
            pred = torch.squeeze(pred)
            l = loss(pred, y)
            optim.zero_grad()
            l.backward()
            optim.step()
            total_loss += l.item()

        scheduler.step()
        print(f'epoch: {epoch}, accuracy: {evaluate_one_class_classifier(model, validation_dataset, config): .4f}, loss: {total_loss/len(train_dataloader) : .4f}')
    torch.save(model.state_dict(), f'./cifar_binary_{config["class"]}_classifier')


def train_encoder(config):
    # transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])

    c = classifier(num_classes=1)
    c.load_state_dict(torch.load('cifar_binary_0_classifier'))
    dataset = EmbeddedDataset(OneClassDataset(CIFAR10(root='../', train=True), one_class_labels=[config['class']], transform=ToTensor()), classifier=c)
    train_dataset = Subset(dataset, range(0, (int)(0.7 * len(dataset))))
    validation_dataset = Subset(dataset, range((int)(0.7 * len(dataset)), len(dataset)))

    f = Discriminator(config['encoding_dim']).cuda()
    e = Encoder(config['encoding_dim']).cuda()

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

def evaluate_classifier(model, validation_dataset, config):

    model.eval()
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
    with torch.no_grad():
        correct = torch.zeros(1).cuda()
        incorrect = torch.zeros(1).cuda()
        for x, y in validation_dataloader:
            x = x.cuda()
            y = y.cuda()
            _, pred = model(x)
            pred = torch.argmax(softmax(pred, dim=1), dim=1)
            correct += torch.sum(y == pred)
            incorrect += torch.sum(y != pred)

        accuracy = (correct/(correct+incorrect)).item()

    model.train()

    return accuracy

def evaluate_one_class_classifier(model, validation_dataset, config):

    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], num_workers=20)

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        correct = torch.tensor([0]).cuda()
        incorrect = torch.tensor([0]).cuda()
        for x, l in validation_dataloader:
            x = x.cuda()
            l = l.cuda()
            _, y_logit = model(x)
            y_pred = (sigmoid(y_logit) > 0.5)*1
            correct += torch.sum(y_pred == l)
            incorrect += torch.sum(y_pred != l)

        accuracy = (correct/(correct + incorrect)).item()

    model.train()

    return accuracy


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
    config = {'batch_size': 64, 'epochs': 200, 'encoding_dim': 8, 'encoder_iters': 1000000, 'discriminator_n': 4, 'lr': 1e-3, 'weight_decay': 1e-5, 'clip': 1e-2}

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