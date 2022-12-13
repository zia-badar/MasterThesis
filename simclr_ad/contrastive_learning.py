import torch
from sklearn.metrics import roc_auc_score
from torch import nn, norm, optim
from torch.nn.functional import normalize
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import efficientnet_b0, resnet50, resnet18
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from torchlars import LARS

from dataset import OneClassDataset, AugmentedDataset


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet18().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return feature, out

# class efficient_net(nn.Module):
#
#     def __init__(self, config):
#         super(efficient_net, self).__init__()
#
#         model = efficientnet_b0()
#         filtered_child_modules = []
#         for name, child_module in model.named_children():
#             if name != 'classifier':
#                 filtered_child_modules.append(child_module)
#         filtered_child_modules.append(nn.Flatten())
#
#         # features
#         self.f = nn.Sequential(*filtered_child_modules)
#
#         # projected features
#         self.g = nn.Sequential(
#             nn.Linear(1280, 512),
#             nn.BatchNorm1d(num_features=512),
#             nn.Linear(in_features=512, out_features=config['feature_projection_dim'])
#         )
#
#     def __call__(self, x):
#         f = self.f(x)
#         g = self.g(f)
#
#         return f, g

# https://arxiv.org/pdf/2004.11362.pdf
def contrastive_loss(z, z_aug):
    z = normalize(z, dim=1)
    z_aug = normalize(z_aug, dim=1)

    batch_size = z.shape[0]
    temperature = 0.5

    # simclr
    pos = torch.exp((z[:, None, :] @ z_aug[:, :, None]).squeeze()/temperature)
    mask = torch.cat([torch.logical_not(torch.eye(batch_size).cuda()), torch.logical_not(torch.eye(batch_size).cuda())])
    neg = torch.sum(torch.masked_select(torch.exp((z @ torch.cat([z, z_aug]).T) / temperature), mask.T).view(batch_size, 2*batch_size-2), dim=1)
    l = -torch.mean(torch.log(pos/(pos + neg)))

    return l

def evaluated(model, train_dataset, validation_dataset, config):

    model.eval()
    with torch.no_grad():
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
        zs = []
        for x, _ in train_dataloader:
            _, z = model(x.cuda())
            z = normalize(z, dim=1)
            zs.append(z)

        zs = torch.cat(zs)

        validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)
        scores = []
        labels = []
        for x, l in validation_dataloader:
            _, z = model(x.cuda())
            scores.append(torch.max(z @ zs.T, dim=1).values)
            labels.append(l)

        scores = torch.cat(scores).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        roc = roc_auc_score(labels, scores)

    model.train()

    return roc

def train(config):
    inlier_dataset = OneClassDataset(CIFAR10(root='', train=True), one_class_labels=[config['class']])
    train_dataset = Subset(inlier_dataset, range(0, (int)(0.7*len(inlier_dataset))))

    validation_inlier_dataset = Subset(inlier_dataset, range((int)(0.7*len(inlier_dataset)), len(inlier_dataset)))
    outlier_classes = list(range(10))
    outlier_classes.remove(config['class'])
    outlier_dataset = OneClassDataset(CIFAR10(root='', train=True), zero_class_labels=outlier_classes)
    validation_dataset = ConcatDataset([validation_inlier_dataset, outlier_dataset])
    validation_dataset = AugmentedDataset(validation_dataset, pair=False)
    without_pair_train_dataset = AugmentedDataset(train_dataset, pair=False)

    train_dataset = AugmentedDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=20)

    # model = efficient_net(config).cuda()
    model = Model().cuda()
    base_optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-6)
    optim = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    scheduler = CosineAnnealingLR(optim, config['epochs'])
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=10.0, total_epoch=10, after_scheduler=scheduler)

    model.train()

    for epoch in range(1, config['epochs']+1):
        total_loss = 0
        for i, (x, x_aug, _) in enumerate(train_dataloader):
            _, z = model(x.cuda())
            _, z_aug = model(x_aug.cuda())

            optim.zero_grad()
            loss = contrastive_loss(z, z_aug)
            loss.backward()
            optim.step()
            total_loss += loss.item()

            scheduler_warmup.step(epoch - 1 + i / len(train_dataloader))


        # progress_bar.set_description(f'loss: {total_loss: .4f}')
        print(f'loss: {total_loss/len(train_dataloader): .2f}, epoch: {epoch}')
        if epoch % 50 == 0:
            print(f'roc: {evaluated(model, without_pair_train_dataset, validation_dataset, config)}')
            if epoch % 100 == 0:
                torch.save(model.state_dict(), 'constrastive_model')

if __name__ == '__main__':
    config = {'class': -1, 'batch_size': 512, 'feature_projection_dim': 128, 'epochs': 10000}

    config['class'] = 0
    train(config)