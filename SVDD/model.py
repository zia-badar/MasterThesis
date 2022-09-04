import torch
from torch import nn
from torch.nn.functional import leaky_relu, interpolate
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, model_type, use_in_autoencoder=True):
        super(Encoder, self).__init__()

        self.model_type = model_type
        self.use_in_autoencoder = use_in_autoencoder
        self.encode_dim = 32 if model_type == 'mnist' else 128

        self.pool = nn.MaxPool2d(2, 2)

        if self.model_type == 'mnist':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=8, eps=1e-4, affine=False)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=4, eps=1e-4, affine=False)
            self.flatten = nn.Flatten()
            self.out = nn.Linear(in_features=4 * 7 * 7, out_features=self.encode_dim, bias=False)
        elif self.model_type == 'cifar':
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-4, affine=False)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-4, affine=False)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=128, eps=1e-4, affine=False)
            self.flatten = nn.Flatten()
            self.out = nn.Linear(in_features=128 * 4 * 4, out_features=self.encode_dim, bias=False)
            self.bn4 = nn.BatchNorm1d(num_features=self.encode_dim, eps=1e-4, affine=False)
            nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))


    def forward(self, batch):
        x = self.conv1(batch)
        x = self.pool(leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(leaky_relu(self.bn2(x)))
        if self.model_type == 'cifar':
            x = self.conv3(x)
            x = self.pool(leaky_relu(self.bn3(x)))
        x = self.flatten(x)
        x = self.out(x)
        if self.model_type == 'cifar' and self.use_in_autoencoder:      # donot batch normalize when encoder is used in Deep SVDD
            x = self.bn4(x)
        return x

class Decoder(nn.Module):

    def __init__(self, model_type):
        super(Decoder, self).__init__()
        self.model_type = model_type
        self.encode_dim = 32 if model_type == 'mnist' else 128

        if self.model_type == 'mnist':
            self.deconv1 = nn.ConvTranspose2d(in_channels=(int)(self.encode_dim / (4*4)), out_channels=4, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=4, eps=1e-4, affine=False)
            self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=(5, 5), padding=(3, 3), bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=8, eps=1e-4, affine=False)
            self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=(5, 5), padding=(2, 2), bias=False)
        elif self.model_type == 'cifar':
            self.deconv1 = nn.ConvTranspose2d(in_channels=(int)(self.encode_dim / (4 * 4)), out_channels=128, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=128, eps=1e-4, affine=False)
            self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-4, affine=False)
            self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=(2, 2), bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-4, affine=False)
            self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5), padding=(2, 2), bias=False)
            nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
            nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, batch):
        x = batch
        x = leaky_relu(x)
        if self.model_type == 'mnist':
            x = interpolate(x, scale_factor=2)
        x = self.deconv1(x)
        x = interpolate(leaky_relu(self.bn1(x)), scale_factor=2)
        x = self.deconv2(x)
        x = interpolate(leaky_relu(self.bn2(x)), scale_factor=2)
        x = self.deconv3(x)
        if self.model_type == 'cifar':
            x = interpolate(leaky_relu(self.bn3(x)), scale_factor=2)
            x = self.deconv4(x)

        x = torch.sigmoid(x)                # because input is min-max normalized
        return x