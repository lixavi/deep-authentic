import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, image_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.image_size * self.image_size * self.channels),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels, self.image_size, self.image_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, image_size, channels):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channels = channels

        self.model = nn.Sequential(
            nn.Linear(self.image_size * self.image_size * self.channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
