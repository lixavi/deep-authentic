import torch
from torchvision.utils import save_image

def test_gan(generator, num_samples, latent_dim, device, output_dir):
    generator.eval()

    # Generate samples from the trained generator
    with torch.no_grad():
        latent_vectors = torch.randn(num_samples, latent_dim).to(device)
        generated_images = generator(latent_vectors).cpu()

    # Save generated images
    for i in range(num_samples):
        save_image(generated_images[i], f"{output_dir}/generated_image_{i+1}.png")

    print(f"Generated {num_samples} images saved in {output_dir}")
   