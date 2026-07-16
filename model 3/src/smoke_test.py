# src/smoke_test.py (sección de benchmark actualizada)
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model import get_model
from preprocessing import train_dataset

# Acelera cuDNN para tamaño de input fijo (384x384 siempre) — se auto-optimiza tras el warmup
torch.backends.cudnn.benchmark = True


def get_balanced_subset_indices(dataset, n_per_class, seed=42):
    random.seed(seed)
    class_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    selected = []
    for label, indices in class_indices.items():
        selected.extend(random.sample(indices, min(n_per_class, len(indices))))

    random.shuffle(selected)
    return selected


def benchmark_batch_size(batch_size, device, n_warmup=3, n_measure=10):
    """
    Mide throughput real (imágenes/segundo), descartando los primeros
    batches (warmup de cuDNN/CUDA) del cálculo de tiempo.
    """
    # Subset grande y fijo para tener suficientes batches en TODAS las pruebas
    n_per_class = max(100, (n_warmup + n_measure) * batch_size // 4 + 20)
    indices = get_balanced_subset_indices(train_dataset, n_per_class)
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    total_batches_needed = n_warmup + n_measure
    if len(loader) < total_batches_needed:
        print(f"⚠️ batch_size={batch_size}: subset insuficiente ({len(loader)} batches disponibles, "
              f"se necesitan {total_batches_needed}). Saltando.")
        return None

    model = get_model(num_classes=4, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    try:
        for i, (images, labels) in enumerate(loader):
            if i >= total_batches_needed:
                break

            images, labels = images.to(device), labels.to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            if i >= n_warmup:  # descarta los primeros batches (warmup)
                times.append(t1 - t0)

    except torch.cuda.OutOfMemoryError:
        print(f"❌ OUT OF MEMORY con batch_size={batch_size}")
        return None

    avg_time = sum(times) / len(times)
    images_per_sec = batch_size / avg_time
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

    print(f"batch_size={batch_size:3d} | tiempo/batch (estable)={avg_time:.3f}s | "
          f"img/s={images_per_sec:.1f} | memoria={peak_mem:.2f}GB")

    return {"batch_size": batch_size, "avg_time": avg_time,
            "images_per_sec": images_per_sec, "peak_memory_gb": peak_mem}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device} | GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}\n")

    results = []
    for bs in [16, 32, 48, 64, 96]:
        result = benchmark_batch_size(bs, device)
        if result is None:
            continue
        results.append(result)
        if result["peak_memory_gb"] > 3.5:  # margen de seguridad antes de OOM
            print("Memoria acercándose al límite, deteniendo pruebas.")
            break

    print(f"\n{'='*60}")
    print("RESUMEN (ordenado por throughput)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x["images_per_sec"]):
        print(f"batch_size={r['batch_size']:3d} | img/s={r['images_per_sec']:6.1f} | "
              f"memoria={r['peak_memory_gb']:.2f}GB")


if __name__ == "__main__":
    main()