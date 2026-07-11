from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

IMG_SIZE = 224
BATCH_SIZE = 32

def rgb_loader(path):
    return Image.open(path).convert("RGB")

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir='model 3/data'):
    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms, loader=rgb_loader)
    val_ds   = datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transforms, loader=rgb_loader)
    test_ds  = datasets.ImageFolder(f"{data_dir}/test", transform=val_test_transforms, loader=rgb_loader)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, train_ds.classes