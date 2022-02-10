import torch
import torchvision.transforms as transforms
from PIL import Image

class DeepFakeSynthesizer:
    def __init__(self, generator_model_path):
        self.generator = self.load_generator(generator_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_generator(self, generator_model_path):
        # Load pre-trained GAN generator model
        generator = torch.load(generator_model_path, map_location=self.device)
        generator.eval()
        return generator



        # Generate DeepFake using GAN generator
        with torch.no_grad():
            generated_image = self.generator(face_tensor).squeeze(0).cpu()

        # Convert generated tensor back to PIL image
        generated_image = transforms.ToPILImage()(generated_image)

        return generated_image
