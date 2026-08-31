import rasterio
import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# 1. Chemins des images
# ==================================================

image_2020 = "data/raw/2020/sentinel2_manaus_2020.tif"
image_2024 = "data/raw/2024/sentinel2_manaus_2024.tif"


# ==================================================
# 2. Lecture des bandes
# ==================================================

with rasterio.open(image_2020) as src:

    red_2020 = src.read(3).astype("float32")
    nir_2020 = src.read(4).astype("float32")


with rasterio.open(image_2024) as src:

    red_2024 = src.read(3).astype("float32")
    nir_2024 = src.read(4).astype("float32")


# ==================================================
# 3. Masques de validité
# ==================================================

valid_2020 = (
    (red_2020 > 0) &
    (nir_2020 > 0)
)

valid_2024 = (
    (red_2024 > 0) &
    (nir_2024 > 0)
)


# On ne garde que les pixels valides
# pour les deux années.

valid = valid_2020 & valid_2024


print("Pixels communs valides :", np.sum(valid))


# ==================================================
# 4. Calcul NDVI 2020
# ==================================================

ndvi_2020 = np.full(
    red_2020.shape,
    np.nan,
    dtype="float32"
)

denom_2020 = nir_2020 + red_2020

mask_2020 = valid & (denom_2020 != 0)

ndvi_2020[mask_2020] = (
    (nir_2020[mask_2020] - red_2020[mask_2020])
    /
    denom_2020[mask_2020]
)


# ==================================================
# 5. Calcul NDVI 2024
# ==================================================

ndvi_2024 = np.full(
    red_2024.shape,
    np.nan,
    dtype="float32"
)

denom_2024 = nir_2024 + red_2024

mask_2024 = valid & (denom_2024 != 0)

ndvi_2024[mask_2024] = (
    (nir_2024[mask_2024] - red_2024[mask_2024])
    /
    denom_2024[mask_2024]
)


# ==================================================
# 6. Différence NDVI
# ==================================================

delta_ndvi = ndvi_2024 - ndvi_2020


# ==================================================
# 7. Statistiques
# ==================================================

print("\n===== DIFFERENCE NDVI =====")

print(
    "Minimum :",
    np.nanmin(delta_ndvi)
)

print(
    "Maximum :",
    np.nanmax(delta_ndvi)
)

print(
    "Moyenne :",
    np.nanmean(delta_ndvi)
)


# ==================================================
# 8. Détection des fortes pertes
# ==================================================

# Seuil provisoire de baseline
# Une diminution supérieure à 0.20
# est considérée comme une perte importante
# de végétation.

deforestation_mask = (
    delta_ndvi < -0.20
)


print(
    "\nPixels avec forte diminution :",
    np.sum(deforestation_mask)
)


# ==================================================
# 9. Visualisation du delta NDVI
# ==================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    delta_ndvi,
    vmin=-0.5,
    vmax=0.5
)

plt.colorbar(
    label="ΔNDVI (2024 - 2020)"
)

plt.title(
    "Variation du NDVI — Manaus — 2020 → 2024"
)

plt.axis("off")

plt.savefig(
    "reports/delta_ndvi_manaus_2020_2024.png",
    bbox_inches="tight",
    dpi=200
)

plt.close()


# ==================================================
# 10. Carte de déforestation potentielle
# ==================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    deforestation_mask,
    vmin=0,
    vmax=1
)

plt.title(
    "Zones de forte diminution de végétation"
)

plt.axis("off")

plt.savefig(
    "reports/potential_deforestation_manaus.png",
    bbox_inches="tight",
    dpi=200
)

plt.close()


print(
    "\nImages sauvegardées dans reports/"
)
