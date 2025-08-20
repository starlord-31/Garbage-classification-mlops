import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int, device: torch.device):
    """
    Initialize a pretrained model with the final classification layer adjusted
    for the specified number of classes.
    Features are frozen for transfer learning.
    Supports: effnetb0, mobilenetv3, resnet18, densenet121, vit_b_16.
    """

    if model_name == "effnetb0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "mobilenetv3":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(960, in_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":  # Vision Transformer
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        for param in model.encoder.parameters():
            param.requires_grad = False
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    print(f"[INFO] Loaded model '{model_name}' on device {device}")
    return model


if __name__ == "__main__":
    # Demo usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6  # Replace with actual number of classes

    model_names = [
        "effnetb0",
        "mobilenetv3",
        "resnet18",
        "densenet121",
        "vit_b_16",
    ]
    models_dict = {}

    for name in model_names:
        models_dict[name] = get_model(name, num_classes, device)

    for name, model in models_dict.items():
        classifier_part = (
            model.classifier
            if hasattr(model, "classifier")
            else model.fc if hasattr(model, "fc") else model.heads.head
        )

        print(f"Model {name} classifier structure:\n{classifier_part}\n")
