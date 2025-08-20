import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count() if os.cpu_count() is not None else 2


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_transform: transforms.Compose,
    val_test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    train_data = datasets.ImageFolder(
        str(train_dir), transform=train_transform
    )
    val_data = datasets.ImageFolder(str(val_dir), transform=val_test_transform)
    test_data = datasets.ImageFolder(
        str(test_dir), transform=val_test_transform
    )

    # Get classes from train data
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader, test_dataloader, class_names


# Example usage:
if __name__ == "__main__":
    from torchvision.models import EfficientNet_B0_Weights

    # Set paths to split data
    base_path = Path("subset_data")
    train_dir = base_path / "train"
    val_dir = base_path / "val"
    test_dir = base_path / "test"

    # Transforms (reuse consistent config from your prep script)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    BATCH_SIZE = 32

    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_transform=train_transforms,
        val_test_transform=base_transforms,
        batch_size=BATCH_SIZE,
    )
    print(f"Number of training batches: {len(train_dl)}")
    print(f"Number of validation batches: {len(val_dl)}")
    print(f"Number of testing batches: {len(test_dl)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
