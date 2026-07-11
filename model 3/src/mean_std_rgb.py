from pathlib import Path

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# CALCS MADE AFTER CONFIRMING RGB: 4217

# Transformaciones temporales:
# Solo convertimos a RGB, redimensionamos y pasamos a tensor.
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="data/raw",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

mean = torch.zeros(3)
std = torch.zeros(3)
num_images = 0

for images, _ in loader:

    # images: [batch, channels, height, width]

    batch_size = images.size(0)

    images = images.view(batch_size, 3, -1)

    mean += images.mean(dim=2).sum(dim=0)

    std += images.std(dim=2).sum(dim=0)

    num_images += batch_size

mean /= num_images
std /= num_images

print("\nMedia:")
print(mean)

print("\nDesviación estándar:")
print(std)