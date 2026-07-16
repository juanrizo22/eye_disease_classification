# src/train.py
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import get_model, unfreeze_backbone, count_parameters
from preprocessing import train_loader, validation_loader

torch.backends.cudnn.benchmark = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train() if train else model.eval()
    running_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            # Mixed precision: reduce uso de memoria y acelera en GPU moderna
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_phase(model, train_loader, val_loader, optimizer, epochs,
                 patience, phase_name, scheduler=None):
    """
    Entrena una fase completa con early stopping.
    Retorna el modelo con los mejores pesos según val_loss.
    """
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, scaler, DEVICE, train=False)

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"[{phase_name}] Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping basado en val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_{phase_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en epoch {epoch+1} (sin mejora en {patience} épocas)")
                break

    model.load_state_dict(best_weights)
    return model, history


def main():
    print(f"Usando dispositivo: {DEVICE}")

    model = get_model(num_classes=4, freeze_backbone=True)
    model.to(DEVICE)

    # ============================================================
    # FASE 1 — Backbone congelado, solo entrena la cabeza custom
    # ============================================================
    print("\n=== FASE 1: entrenando solo la cabeza (backbone congelado) ===")
    count_parameters(model)

    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1, mode="min", factor=0.5, patience=2
    )

    model, history1 = train_phase(
        model, train_loader, validation_loader,
        optimizer=optimizer_phase1,
        epochs=10,
        patience=4,
        phase_name="phase1_head",
        scheduler=scheduler_phase1
    )

    # ============================================================
    # FASE 2 — Descongelar backbone completo, fine-tuning fino
    # ============================================================
    print("\n=== FASE 2: fine-tuning del backbone completo ===")
    model = unfreeze_backbone(model, num_layers_to_unfreeze=None)  # descongela todo
    count_parameters(model)

    # Learning rates diferenciados: backbone MUY bajo, cabeza moderado
    # Evita destruir los pesos preentrenados con gradientes grandes
    optimizer_phase2 = optim.Adam([
        {"params": model.features.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ])
    scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2, mode="min", factor=0.5, patience=2
    )

    model, history2 = train_phase(
        model, train_loader, validation_loader,
        optimizer=optimizer_phase2,
        epochs=20,
        patience=5,
        phase_name="phase2_finetune",
        scheduler=scheduler_phase2
    )

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pth")
    print("\nEntrenamiento completo. Modelo final guardado en checkpoints/final_model.pth")

    return model, history1, history2


if __name__ == "__main__":
    main()