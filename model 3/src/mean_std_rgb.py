# mean_std_rgb.py (CORREGIDO)
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def rgb_loader(path):
    return Image.open(path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((384, 384)),   # usa el tamaño final que vas a entrenar, no 256
    transforms.ToTensor()
])

# CLAVE: apunta a data/split/train, NO a data/raw
PROJECT_ROOT = Path(__file__).resolve().parents[1]
train_dir = PROJECT_ROOT / "data" / "split" / "train"

dataset = datasets.ImageFolder(root=train_dir, transform=transform, loader=rgb_loader)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

sum_ = torch.zeros(3)
sum_sq = torch.zeros(3)
num_pixels = 0

for images, _ in loader:
    b, c, h, w = images.shape
    images = images.view(b, c, -1)
    sum_ += images.sum(dim=[0, 2])
    sum_sq += (images ** 2).sum(dim=[0, 2])
    num_pixels += b * h * w

mean = sum_ / num_pixels
std = torch.sqrt(sum_sq / num_pixels - mean ** 2)

print("\nMedia:", mean)
print("Desviación estándar:", std)