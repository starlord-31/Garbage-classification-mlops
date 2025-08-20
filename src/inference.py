import torch
import mlflow
import mlflow.pytorch
from mlflow.entities import ViewType
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path
from create_dataloaders import create_dataloaders
from torchvision.models import EfficientNet_B0_Weights


def get_best_run_id(metric_name='val_acc', experiment_id=None):
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs([experiment_id], order_by=[f"metrics.{metric_name} DESC"])
    best_run_id = runs[0].info.run_id if runs else None
    best_val_acc = runs[0].data.metrics[metric_name] if runs else None
    return best_run_id, best_val_acc


def load_best_mlflow_model(best_run_id: str, device: torch.device) -> torch.nn.Module:
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from MLflow run {best_run_id}")
    return model


def pred_and_plot_images(
    model: torch.nn.Module,
    test_dl: torch.utils.data.DataLoader,
    class_names: list,
    device: torch.device,
    num_examples: int = 10
):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 8))
    for images, labels in test_dl:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        preds_idx = preds.argmax(dim=1)
        img_grid = make_grid(images.cpu(), nrow=min(num_examples, len(images)), normalize=True)
        plt.imshow(img_grid.permute(1, 2, 0))
        title_labels = [class_names[i] for i in preds_idx.cpu().numpy()]
        plt.title("Predictions: " + " | ".join(title_labels))
        plt.axis("off")
        plt.show()
        images_shown += images.size(0)
        if images_shown >= num_examples:
            break


def get_experiment_id_from_mlruns(mlruns_path):
    mlruns_dir = Path(mlruns_path)
    experiment_ids = [p.name for p in mlruns_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    if not experiment_ids:
        raise ValueError(f"No experiment folders found in {mlruns_path}")
    # For demonstration, pick the first experiment ID found
    return experiment_ids[0]


if __name__ == "__main__":
    mlruns_path = "/home/starlord/Garbage-classification-mlops/mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    client = mlflow.tracking.MlflowClient()

    experiment_id = get_experiment_id_from_mlruns(mlruns_path)
    print(f"Using experiment ID from mlruns folder: {experiment_id}")

    metric_name = 'val_acc'
    best_run_id, best_val_acc = get_best_run_id(metric_name, experiment_id)

    if best_run_id is not None and best_val_acc is not None:
        print(f"Best run ID: {best_run_id} (val_acc={best_val_acc:.4f})")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model = load_best_mlflow_model(best_run_id, device)
    else:
        print(f"No valid runs found in experiment {experiment_id}")
        exit(1)

    # Get DataLoader and class names
    base_path = Path("subset_data")
    test_dir = base_path / "test"
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()
    BATCH_SIZE = 10
    _, _, test_dl, class_names = create_dataloaders(
        train_dir=base_path / "train",
        val_dir=base_path / "val",
        test_dir=test_dir,
        train_transform=base_transforms,
        val_test_transform=base_transforms,
        batch_size=BATCH_SIZE
    )

    pred_and_plot_images(
        model=best_model,
        test_dl=test_dl,
        class_names=class_names,
        device=device,
        num_examples=10
    )
