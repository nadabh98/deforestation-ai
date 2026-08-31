import rasterio
import numpy as np
import matplotlib.pyplot as plt


def calculate_ndvi(image_path, output_path, title):

    # ---------------------------------------------
    # 1. Lecture du GeoTIFF
    # ---------------------------------------------

    with rasterio.open(image_path) as src:

        red = src.read(3).astype("float32")
        nir = src.read(4).astype("float32")

    # ---------------------------------------------
    # 2. Masque des pixels valides
    # ---------------------------------------------

    valid_mask = (
        (red > 0) &
        (nir > 0)
    )

    print("\nImage :", image_path)
    print("Pixels valides :", np.sum(valid_mask))
    print("Pixels invalides :", np.sum(~valid_mask))

    # ---------------------------------------------
    # 3. Calcul du NDVI
    # ---------------------------------------------

    ndvi = np.full(
        red.shape,
        np.nan,
        dtype="float32"
    )

    denominator = nir + red

    valid_calculation = (
        valid_mask &
        (denominator != 0)
    )

    ndvi[valid_calculation] = (
        (nir[valid_calculation] - red[valid_calculation])
        /
        denominator[valid_calculation]
    )

    # ---------------------------------------------
    # 4. Statistiques
    # ---------------------------------------------

    print(
        "NDVI minimum :",
        np.nanmin(ndvi)
    )

    print(
        "NDVI maximum :",
        np.nanmax(ndvi)
    )

    print(
        "NDVI moyen :",
        np.nanmean(ndvi)
    )

    # ---------------------------------------------
    # 5. Visualisation
    # ---------------------------------------------

    plt.figure(figsize=(10, 10))

    plt.imshow(
        ndvi,
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        label="NDVI"
    )

    plt.title(title)

    plt.axis("off")

    # ---------------------------------------------
    # 6. Sauvegarde
    # ---------------------------------------------

    plt.savefig(
        output_path,
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    print(
        "Image sauvegardée :",
        output_path
    )


# =============================================
# 2020
# =============================================

calculate_ndvi(
    "data/raw/2020/sentinel2_manaus_2020.tif",
    "reports/ndvi_manaus_2020.png",
    "NDVI — Manaus — 2020"
)


# =============================================
# 2024
# =============================================

calculate_ndvi(
    "data/raw/2024/sentinel2_manaus_2024.tif",
    "reports/ndvi_manaus_2024.png",
    "NDVI — Manaus — 2024"
)
