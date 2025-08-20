import os
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path
from model import get_model  # Your model getter function


def save_model(model, target_dir, model_name):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_name
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device, writer=None):
    model.to(device)
    best_val_acc = 0.0
    avg_train_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += inputs.size(0)

        avg_train_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        model.eval()
        val_correct = 0
        val_examples = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_examples += inputs.size(0)
        val_acc = val_correct / val_examples if val_examples else 0.0

        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss={avg_train_loss:.4f} Train Acc={train_acc:.4f} Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return {"val_acc": best_val_acc, "train_loss": avg_train_loss}


def run_series_of_experiments(num_epochs_list, model_names, train_dl, val_dl, num_classes, device, results_dir="models"):
    mlflow.set_tracking_uri("file:///home/starlord/Garbage-classification-mlops/mlruns")
    mlflow.set_experiment("default")

    experiment_number = 0
    for epochs in num_epochs_list:
        for model_name in model_names:
            experiment_number += 1
            print(f"[INFO] Experiment number: {experiment_number}")
            print(f"[INFO] Model: {model_name}")
            print(f"[INFO] Number of epochs: {epochs}")

            model = get_model(model_name, num_classes, device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
            writer = None
            example_input = torch.randn(1, 3, 224, 224).to(device)

            with mlflow.start_run(run_name=f"{model_name}_{epochs}epochs"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("device", str(device))
                mlflow.log_param("num_classes", num_classes)

                training_results = train(
                    model=model,
                    train_dataloader=train_dl,
                    val_dataloader=val_dl,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=epochs,
                    device=device,
                    writer=writer
                )

                for k, v in training_results.items():
                    if isinstance(v, (float, int)):
                        mlflow.log_metric(k, v)

                print("MLflow Tracking URI:", mlflow.get_tracking_uri())
                print("MLflow Artifact URI env var:", os.getenv("MLFLOW_ARTIFACT_URI"))
                print("Current working directory:", os.getcwd())

                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    input_example=example_input.cpu().numpy()
                )

            filepath = f"GIC_{model_name}_{epochs}_epochs.pth"
            save_model(model=model, target_dir=results_dir, model_name=filepath)
            print("-" * 50 + "\n")


if __name__ == "__main__":
    from create_dataloaders import create_dataloaders

    num_epochs = [5, 10, 15]
    model_names = ["effnetb0", "mobilenetv3", "resnet18", "densenet121", "vit_b_16"]
    base_path = Path("subset_data")
    train_dir = base_path / "train"
    val_dir = base_path / "val"
    batch_size = 32

    # Assumes EfficientNet_B0_Weights from torchvision is available
    from torchvision.models import EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transforms = weights.transforms()
    train_transforms = base_transforms

    train_dl, val_dl, _, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=None,
        train_transform=train_transforms,
        val_test_transform=base_transforms,
        batch_size=batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)

    run_series_of_experiments(
        num_epochs_list=num_epochs,
        model_names=model_names,
        train_dl=train_dl,
        val_dl=val_dl,
        num_classes=num_classes,
        device=device,
        results_dir="models",
    )
