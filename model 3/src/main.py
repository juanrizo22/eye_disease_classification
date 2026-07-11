# main.py
from dataset import get_dataloaders

def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders()

    print("Clases:", classes)
    print("Batches train/val/test:", len(train_loader), len(val_loader), len(test_loader))

    images, labels = next(iter(train_loader))
    print("Shape imágenes:", images.shape)
    print("Shape labels:", labels.shape)

if __name__ == "__main__":
    main()