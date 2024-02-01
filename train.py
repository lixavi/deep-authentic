import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, device):
    adversarial_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_images in enumerate(tqdm(dataloader)):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            discriminator_optimizer.zero_grad()

            # Adversarial loss for real images
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_images)
            d_loss_real = adversarial_loss(real_output, real_labels)

            # Adversarial loss for fake images
            latent_vectors = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(latent_vectors)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = adversarial_loss(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            discriminator_optimizer.step()

            # Train Generator
            generator_optimizer.zero_grad()
            fake_labels.fill_(1)
            fake_output = discriminator(fake_images)
            g_loss = adversarial_loss(fake_output, fake_labels)
            g_loss.backward()
            generator_optimizer.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
