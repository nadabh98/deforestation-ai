import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PARAMÈTRES
# ============================================================

IMAGE_2020 = "data/raw/2020/sentinel2_manaus_2020.tif"
MASK_DIR = "data/processed/masks"

OUTPUT = "reports/deforestation_mask_overview.png"


# ============================================================
# LECTURE DE L'IMAGE 2020
# ============================================================

with rasterio.open(IMAGE_2020) as src:

    height = src.height
    width = src.width

    red = src.read(3).astype(np.float32)
    nir = src.read(4).astype(np.float32)

    transform = src.transform

    print("Dimensions :", width, "x", height)


# ============================================================
# CALCUL DU NDVI
# ============================================================

denominator = nir + red

valid = (
    (red > 0) &
    (nir > 0) &
    (denominator > 0)
)

ndvi = np.full(
    red.shape,
    np.nan,
    dtype=np.float32
)

ndvi[valid] = (
    (nir[valid] - red[valid])
    /
    denominator[valid]
)


# ============================================================
# RECONSTRUCTION DU MASQUE
# ============================================================

mask = np.zeros(
    (height, width),
    dtype=np.uint8
)

count = 0

for filename in os.listdir(MASK_DIR):

    if not filename.endswith(".tif"):
        continue

    path = os.path.join(
        MASK_DIR,
        filename
    )

    # Extraction des coordonnées
    # Exemple :
    # mask_x1024_y2048.tif

    try:

        x = int(
            filename.split("_x")[1]
            .split("_y")[0]
        )

        y = int(
            filename.split("_y")[1]
            .split(".")[0]
        )

    except (IndexError, ValueError):

        continue


    with rasterio.open(path) as src:

        tile = src.read(1)

        tile_height, tile_width = tile.shape


    # Protection contre les dépassements

    x_end = min(
        x + tile_width,
        width
    )

    y_end = min(
        y + tile_height,
        height
    )

    mask[
        y:y_end,
        x:x_end
    ] = tile[
        0:y_end-y,
        0:x_end-x
    ]

    count += 1


print("Masques chargés :", count)


# ============================================================
# STATISTIQUES
# ============================================================

valid_mask = valid

deforestation_pixels = np.sum(
    (mask == 1) &
    valid_mask
)

valid_pixels = np.sum(
    valid_mask
)

percentage = (
    deforestation_pixels /
    valid_pixels *
    100
)

print()
print("======================================")
print("STATISTIQUES")
print("======================================")

print(
    "Pixels valides :",
    valid_pixels
)

print(
    "Pixels déforestation :",
    deforestation_pixels
)

print(
    "Pourcentage :",
    round(percentage, 2),
    "%"
)


# ============================================================
# VISUALISATION
# ============================================================

plt.figure(
    figsize=(14, 10)
)

# NDVI comme fond

plt.imshow(
    ndvi,
    cmap="Greens",
    vmin=0,
    vmax=1
)


# Masque de déforestation

deforestation = np.ma.masked_where(
    mask == 0,
    mask
)

plt.imshow(
    deforestation,
    cmap="Reds",
    alpha=0.7
)


plt.title(
    "Détection potentielle de déforestation\n"
    "Manaus — 2020 → 2024"
)

plt.axis("off")

plt.tight_layout()


# ============================================================
# SAUVEGARDE
# ============================================================

os.makedirs(
    "reports",
    exist_ok=True
)

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print()
print(
    "Image sauvegardée :",
    OUTPUT
)
