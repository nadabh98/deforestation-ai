import os
import glob

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MASK_DIR = "data/processed/masks"
OUTPUT_DIR = "reports"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "deforestation_map_final.png"
)


# ============================================================
# 1. RECHERCHE DES MASQUES
# ============================================================

mask_files = sorted(
    glob.glob(os.path.join(MASK_DIR, "*.tif"))
)

if len(mask_files) == 0:
    print("Aucun masque trouvé.")
    print(f"Dossier recherché : {MASK_DIR}")
    exit()

print("======================================")
print("CARTE DE DÉFORESTATION")
print("======================================")
print()
print("Nombre de masques :", len(mask_files))


# ============================================================
# 2. LECTURE DES INFORMATIONS DE LA PREMIÈRE TUILE
# ============================================================

with rasterio.open(mask_files[0]) as src:

    tile_height = src.height
    tile_width = src.width
    transform = src.transform
    crs = src.crs

print("Taille d'une tuile :", tile_width, "x", tile_height)
print("CRS :", crs)


# ============================================================
# 3. DÉTERMINATION DES POSITIONS DES TUILES
# ============================================================

tiles = []

max_x = 0
max_y = 0

for file in mask_files:

    filename = os.path.basename(file)

    # Exemple :
    # manaus_2020_x1024_y2048_mask.tif

    parts = filename.replace(".tif", "").split("_")

    x = None
    y = None

    for part in parts:

        if part.startswith("x"):
            try:
                x = int(part[1:])
            except ValueError:
                pass

        if part.startswith("y"):
            try:
                y = int(part[1:])
            except ValueError:
                pass

    if x is None or y is None:
        print("Position impossible à déterminer :", filename)
        continue

    tiles.append((x, y, file))

    max_x = max(max_x, x)
    max_y = max(max_y, y)


# ============================================================
# 4. CRÉATION DE L'IMAGE GLOBALE
# ============================================================

global_width = max_x + tile_width
global_height = max_y + tile_height

print()
print("Dimensions globales :", global_width, "x", global_height)

global_mask = np.zeros(
    (global_height, global_width),
    dtype=np.uint8
)


# ============================================================
# 5. ASSEMBLAGE DES MASQUES
# ============================================================

print()
print("Assemblage des masques...")

for i, (x, y, file) in enumerate(tiles):

    with rasterio.open(file) as src:

        mask = src.read(1)

        h, w = mask.shape

        global_mask[
            y:y+h,
            x:x+w
        ] = mask

    if (i + 1) % 100 == 0:
        print(
            "Masques assemblés :",
            i + 1
        )


# ============================================================
# 6. STATISTIQUES
# ============================================================

pixels_deforestation = np.sum(
    global_mask == 1
)

pixels_total = global_mask.size

pourcentage = (
    pixels_deforestation /
    pixels_total
) * 100

print()
print("======================================")
print("STATISTIQUES")
print("======================================")

print(
    "Pixels déforestation :",
    pixels_deforestation
)

print(
    "Pixels totaux :",
    pixels_total
)

print(
    "Pourcentage :",
    round(pourcentage, 2),
    "%"
)


# ============================================================
# 7. VISUALISATION
# ============================================================

plt.figure(
    figsize=(14, 10)
)

plt.imshow(
    global_mask,
    cmap="Reds",
    interpolation="nearest"
)

plt.title(
    "Détection potentielle de déforestation - Manaus\n"
    f"2020 → 2024 | {pourcentage:.2f}% des pixels détectés"
)

plt.xlabel("Position X")
plt.ylabel("Position Y")

plt.colorbar(
    label="Déforestation"
)

plt.tight_layout()


# ============================================================
# 8. SAUVEGARDE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print("======================================")
print("TRAITEMENT TERMINÉ")
print("======================================")

print(
    "Carte sauvegardée :",
    OUTPUT_FILE
)
