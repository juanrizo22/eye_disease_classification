from pathlib import Path
from PIL import Image
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Ruta automática relativa al script
DATASET = Path(__file__).resolve().parents[1] / "data" / "raw"

# CONTEOS Y ESTADÍSTICAS
widths = []
heights = []
extensions = Counter()
sizes = Counter()
corruptas = []

# VISUALIZING IMAGES
fig, axes = plt.subplots(4, 4, figsize=(10, 10))

carpetas = [f for f in DATASET.iterdir() if f.is_dir()][:4]

for row, folder in enumerate(carpetas):
    n = len(list(folder.glob("*")))
    print(f"{folder.name}: {n} imágenes en total")

    img_validas = 0  

    for img_path in folder.glob("*"):
        try:
            # Control de corrupción
            with Image.open(img_path) as img_check:
                img_check.verify()
            
            # Si es válida, procesamos sus datos
            img = Image.open(img_path)
            
            widths.append(img.width)
            heights.append(img.height)
            extensions[img_path.suffix.lower()] += 1
            
            # <--- NUEVO: Registramos la combinación exacta de (ancho, alto)
            sizes[(img.width, img.height)] += 1

            # Graficar (máximo 4 por fila)
            if img_validas < 4:
                axes[row, img_validas].imshow(img)
                axes[row, img_validas].set_title(folder.name)
                axes[row, img_validas].axis("off")
                img_validas += 1

        except Exception:
            corruptas.append(img_path)
            
    # Ocultar subplots vacíos si quedan
    for col_vacia in range(img_validas, 4):
        axes[row, col_vacia].axis("off")

print("\n" + "="*30)
print(f"Imágenes corruptas detectadas: {len(corruptas)}")
print("="*30 + "\n")

# IMPRESIÓN DE TAMAÑOS COMBINADOS
print("Frecuencia de Tamaños (Ancho, Alto):")
print(sizes)
print("-" * 30)

print("Different Widths: ", sorted(set(widths)))
print("Different Heights: ", sorted(set(heights)))

# DATA FRAME FOR SIZE STATS
df = pd.DataFrame({
    "width": widths, "height": heights
})
print("\nEstadísticas descriptivas:")
print(df.describe())

# PRINT OF FORMAT CHECK
print("\nFormatos detectados:")
print(extensions)

# PLOTTING
plt.tight_layout()
plt.show()