from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42


for split in ["train", "validation", "test"]:

    for class_folder in RAW_DIR.iterdir():

        if class_folder.is_dir():

            (OUTPUT_DIR / split / class_folder.name).mkdir(
                parents=True,
                exist_ok=True
            )

# Copiar imágenes

for class_folder in RAW_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*"))

    train_imgs, temp_imgs = train_test_split(
        images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),
        random_state=RANDOM_STATE
    )

    for img in train_imgs:

        shutil.copy(img, OUTPUT_DIR/"train"/class_folder.name/img.name)

    for img in val_imgs:

        shutil.copy(img, OUTPUT_DIR/"validation"/class_folder.name/img.name)

    for img in test_imgs:

        shutil.copy(img, OUTPUT_DIR/"test"/class_folder.name/img.name)

print("Dataset dividido correctamente.")