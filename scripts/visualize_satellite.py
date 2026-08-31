import rasterio
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Chemin de l'image
# --------------------------------------------------

image_path = "data/raw/2020/sentinel2_manaus_2020.tif"


# --------------------------------------------------
# 2. Lecture du GeoTIFF
# --------------------------------------------------

with rasterio.open(image_path) as src:

    red = src.read(3).astype("float32")
    green = src.read(2).astype("float32")
    blue = src.read(1).astype("float32")


# --------------------------------------------------
# 3. Création du masque des pixels valides
# --------------------------------------------------

valid_mask = (
    (red > 0) &
    (green > 0) &
    (blue > 0)
)

print("Pixels valides :", np.sum(valid_mask))
print("Pixels invalides :", np.sum(~valid_mask))


# --------------------------------------------------
# 4. Normalisation
# --------------------------------------------------

def normalize(band, mask):

    valid_values = band[mask]

    minimum = np.percentile(valid_values, 2)
    maximum = np.percentile(valid_values, 98)

    normalized = np.zeros_like(band)

    normalized[mask] = np.clip(
        (band[mask] - minimum) /
        (maximum - minimum),
        0,
        1
    )

    return normalized


red = normalize(red, valid_mask)
green = normalize(green, valid_mask)
blue = normalize(blue, valid_mask)


# --------------------------------------------------
# 5. Création RGB
# --------------------------------------------------

rgb = np.dstack((red, green, blue))


# --------------------------------------------------
# 6. Affichage
# --------------------------------------------------

plt.figure(figsize=(10, 10))

plt.imshow(rgb)

plt.title("Sentinel-2 — Manaus — 2020")

plt.axis("off")


# --------------------------------------------------
# 7. Sauvegarde
# --------------------------------------------------

output_path = "reports/sentinel2_manaus_2020_rgb_clean.png"

plt.savefig(
    output_path,
    bbox_inches="tight",
    dpi=150
)

plt.close()

print("Image sauvegardée :", output_path)
