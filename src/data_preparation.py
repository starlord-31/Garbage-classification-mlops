from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import random
import sys
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights

# --- Setup ---
data_dir = Path("data/Garbage_Dataset_Classification/images")
target_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
output_dir = Path('subset_data')
subset_dirs = ['train', 'val', 'test']

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

def create_dirs(base_dir: Path, sub_dirs: list, classes: list):
    for sub in sub_dirs:
        for cls in classes:
            (base_dir / sub / cls).mkdir(parents=True, exist_ok=True)

def copy_files(src_files: list, dst_dir: Path):
    for src in src_files:
        shutil.copy2(src, dst_dir / src.name)

def split_dataset():
    create_dirs(output_dir, subset_dirs, target_classes)
    for cls in target_classes:
        class_dir = data_dir / cls
        images = list(class_dir.glob("*.jpg"))
        if not images:
            print(f"[WARNING!] No Images found in {class_dir}", file=sys.stderr)
            continue

        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)
        copy_files(train, output_dir / 'train' / cls)
        copy_files(val, output_dir / 'val' / cls)
        copy_files(test, output_dir / 'test' / cls)
        print(f"Class {cls}\n Training Samples: {len(train)}\n Validation Samples: {len(val)}\n Testing Samples: {len(test)}\n Total Samples: {len(images)}")

if __name__ == "__main__":
    split_dataset()

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
base_transforms = weights.transforms()
print("Base transforms from EfficientNet:", base_transforms)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    normalize
])
print(train_transforms)
