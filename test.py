import torch
from torchvision.utils import save_image

def test_gan(generator, num_samples, latent_dim, device, output_dir):
    generator.eval()

    # Generate samples from the trained generator
    with torch.no_grad():
        latent_vectors = torch.randn(num_samples, latent_dim).to(device)
        generated_images = generator(latent_vectors).cpu()
