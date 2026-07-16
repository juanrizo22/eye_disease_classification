from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# PREVIOUSLY CALCULATED STATS
MEAN = (0.4221, 0.2807, 0.1715)
STD = (0.2950, 0.2172, 0.1637)

IMG_SIZE = 384
BATCH_SIZE = 48

train_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=10),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])

validation_transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(MEAN, STD)

])

###
train_dataset = ImageFolder(
    root="data/split/train",
    transform=train_transform
)

validation_dataset = ImageFolder(
    root="data/split/validation",
    transform=validation_transform
)

test_dataset = ImageFolder(
    root="data/split/test",
    transform=validation_transform
)
###

# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)