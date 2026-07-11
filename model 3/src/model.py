import torch
import torch.nn as nn
from torchvision import models

def build_classifier_head(in_features, num_classes=4):
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes)
    )

def get_model(num_classes=4, freeze_backbone=True):
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280 para EfficientNetV2-S
    model.classifier = build_classifier_head(in_features, num_classes)

    return model

def unfreeze_backbone(model, num_layers_to_unfreeze=None):
    """
    Descongela el backbone para fine-tuning.
    Si num_layers_to_unfreeze es None, descongela todo.
    Si es un número, descongela solo las últimas N capas (más conservador).
    """
    if num_layers_to_unfreeze is None:
        for param in model.features.parameters():
            param.requires_grad = True
    else:
        layers = list(model.features.children())
        for layer in layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
    return model

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total:,}")
    print(f"Parámetros entrenables: {trainable:,} ({100*trainable/total:.1f}%)")

if __name__ == "__main__":
    model = get_model(num_classes=4, freeze_backbone=True)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # esperado: [2, 4]

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables (fase 1): {trainable_params:,}")
