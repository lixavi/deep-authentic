import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = glob.glob(os.path.join(self.data_dir, '*.jpg'))  # Assuming image files are in JPEG format

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def preprocess_data(data_dir, batch_size, image_size):
    # Define transformations for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataset instance
    dataset = DeepFakeDataset(data_dir, transform=transform)

    # Create DataLoader for batch processing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader
