from prefect import flow, task
from pathlib import Path
import torch

@task
def data_prep_task():
    from data_preparation import split_dataset
    split_dataset()

@task
def dataloader_task(batch_size=32):
    from create_dataloaders import create_dataloaders
    from torchvision.models import EfficientNet_B0_Weights
    base_path = Path("subset_data")
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()
    train_dir = base_path / "train"
    val_dir = base_path / "val"
    test_dir = base_path / "test"
    train_dl, val_dl, test_dl, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        train_transform=base_transforms,
        val_test_transform=base_transforms,
        batch_size=batch_size
    )
    return train_dl, val_dl, test_dl, class_names

@task
def run_experiments_task(train_dl, val_dl, class_names, device, batch_size=32):
    from run_experiments import run_series_of_experiments
    num_epochs = [5, 10, 15]
    model_names = ["effnetb0", "mobilenetv3", "resnet18", "densenet121", "vit_b_16"]
    num_classes = len(class_names)
    run_series_of_experiments(
        num_epochs_list=num_epochs,
        model_names=model_names,
        train_dl=train_dl,
        val_dl=val_dl,
        num_classes=num_classes,
        device=device,
        results_dir="models"
    )

@task
def inference_task(test_dl, class_names, device):
    from inference import get_best_run_id, load_best_mlflow_model, pred_and_plot_images
    experiment_id = '0'
    metric_name = 'val_acc'
    best_run_id, best_val_acc = get_best_run_id(metric_name, experiment_id)
    print(f"Best run ID: {best_run_id} (val_acc={best_val_acc:.4f})")
    best_model = load_best_mlflow_model(best_run_id, device)
    pred_and_plot_images(
        model=best_model,
        test_dl=test_dl,
        class_names=class_names,
        device=device,
        num_examples=10
    )

@flow
def full_ml_pipeline():
    data_prep_task()
    train_dl, val_dl, test_dl, class_names = dataloader_task()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiments_task(train_dl, val_dl, class_names, device)
    inference_task(test_dl, class_names, device)

if __name__ == "__main__":
    full_ml_pipeline()
