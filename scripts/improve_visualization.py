import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# Fonction de normalisation
# --------------------------------------------------

def normalize_band(band, valid_mask):

    valid_values = band[valid_mask]

    low = np.percentile(valid_values, 2)
    high = np.percentile(valid_values, 98)

    normalized = np.zeros_like(band, dtype="float32")

    normalized[valid_mask] = np.clip(
        (band[valid_mask] - low) / (high - low),
        0,
        1
    )

    return normalized


# --------------------------------------------------
# Fonction principale
# --------------------------------------------------

def process_image(input_path, year):

    print("\n====================================")
    print("Traitement :", year)
    print("====================================")

    with rasterio.open(input_path) as src:

        # Sentinel-2
        blue = src.read(1).astype("float32")   # B2
        green = src.read(2).astype("float32")  # B3
        red = src.read(3).astype("float32")    # B4
        nir = src.read(4).astype("float32")    # B8

    # --------------------------------------------------
    # Masque des pixels valides
    # --------------------------------------------------

    valid_mask = (
        (blue > 0) &
        (green > 0) &
        (red > 0) &
        (nir > 0)
    )

    print("Pixels valides :", np.sum(valid_mask))
    print("Pixels invalides :", np.sum(~valid_mask))

    # --------------------------------------------------
    # Normalisation
    # --------------------------------------------------

    blue_n = normalize_band(blue, valid_mask)
    green_n = normalize_band(green, valid_mask)
    red_n = normalize_band(red, valid_mask)
    nir_n = normalize_band(nir, valid_mask)

    # --------------------------------------------------
    # RGB naturel
    #
    # Rouge = B4
    # Vert  = B3
    # Bleu  = B2
    # --------------------------------------------------

    rgb = np.dstack([
        red_n,
        green_n,
        blue_n
    ])

    # Pixels invalides en blanc
    rgb[~valid_mask] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(f"Sentinel-2 — RGB naturel — Manaus {year}")
    plt.axis("off")

    output_rgb = f"reports/rgb_manaus_{year}.png"

    plt.savefig(
        output_rgb,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    print("RGB sauvegardé :", output_rgb)

    # --------------------------------------------------
    # FAUSSE COULEUR
    #
    # Rouge = NIR (B8)
    # Vert  = Rouge (B4)
    # Bleu  = Vert (B3)
    # --------------------------------------------------

    false_color = np.dstack([
        nir_n,
        red_n,
        green_n
    ])

    false_color[~valid_mask] = 1

    plt.figure(figsize=(10, 10))
    plt.imshow(false_color)

    plt.title(
        f"Sentinel-2 — Fausse couleur — Manaus {year}"
    )

    plt.axis("off")

    output_false = (
        f"reports/false_color_manaus_{year}.png"
    )

    plt.savefig(
        output_false,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    print(
        "Fausse couleur sauvegardée :",
        output_false
    )


# ==================================================
# TRAITEMENT 2020
# ==================================================

process_image(
    "data/raw/2020/sentinel2_manaus_2020.tif",
    2020
)


# ==================================================
# TRAITEMENT 2024
# ==================================================

process_image(
    "data/raw/2024/sentinel2_manaus_2024.tif",
    2024
)


print("\n====================================")
print("TRAITEMENT TERMINÉ")
print("====================================")
