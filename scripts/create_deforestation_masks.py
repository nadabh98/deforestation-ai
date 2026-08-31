import os
import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# PARAMÈTRES
# ============================================================

IMAGE_2020 = "data/raw/2020/sentinel2_manaus_2020.tif"
IMAGE_2024 = "data/raw/2024/sentinel2_manaus_2024.tif"

OUTPUT_DIR = "data/processed/masks"

# Une diminution du NDVI supérieure à 0.20
# sera considérée comme une déforestation potentielle.
THRESHOLD = -0.20

TILE_SIZE = 256


# ============================================================
# DOSSIER DE SORTIE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FONCTION NDVI
# ============================================================

def calculate_ndvi(red, nir):
    """
    Calcule le NDVI :

        NDVI = (NIR - RED) / (NIR + RED)
    """

    denominator = nir + red

    ndvi = np.zeros_like(nir, dtype=np.float32)

    valid = denominator > 0

    ndvi[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    return ndvi, valid


# ============================================================
# OUVERTURE DES IMAGES
# ============================================================

print("======================================")
print("DÉTECTION DE DÉFORESTATION")
print("======================================")

print()
print("Image 2020 :", IMAGE_2020)
print("Image 2024 :", IMAGE_2024)
print()

with rasterio.open(IMAGE_2020) as src_2020, \
     rasterio.open(IMAGE_2024) as src_2024:

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    if src_2020.width != src_2024.width:
        raise ValueError(
            "Les largeurs des images sont différentes."
        )

    if src_2020.height != src_2024.height:
        raise ValueError(
            "Les hauteurs des images sont différentes."
        )

    width = src_2020.width
    height = src_2020.height

    print("Dimensions :", width, "x", height)
    print("Taille des tuiles :", TILE_SIZE)
    print()
    print("Seuil NDVI :", THRESHOLD)
    print()

    total_tiles = 0
    total_valid = 0
    total_deforestation = 0


    # ========================================================
    # PARCOURS DES TUILES
    # ========================================================

    for y in range(0, height, TILE_SIZE):

        for x in range(0, width, TILE_SIZE):

            tile_width = min(
                TILE_SIZE,
                width - x
            )

            tile_height = min(
                TILE_SIZE,
                height - y
            )

            window = Window(
                x,
                y,
                tile_width,
                tile_height
            )


            # =================================================
            # LECTURE DES BANDES
            # =================================================
            #
            # Notre GeoTIFF contient :
            #
            # bande 1 = B2
            # bande 2 = B3
            # bande 3 = B4
            # bande 4 = B8
            #
            # Pour NDVI :
            #
            # RED = B4
            # NIR = B8
            # =================================================

            red_2020 = src_2020.read(
                3,
                window=window
            ).astype(np.float32)

            nir_2020 = src_2020.read(
                4,
                window=window
            ).astype(np.float32)

            red_2024 = src_2024.read(
                3,
                window=window
            ).astype(np.float32)

            nir_2024 = src_2024.read(
                4,
                window=window
            ).astype(np.float32)


            # =================================================
            # CALCUL NDVI
            # =================================================

            ndvi_2020, valid_2020 = calculate_ndvi(
                red_2020,
                nir_2020
            )

            ndvi_2024, valid_2024 = calculate_ndvi(
                red_2024,
                nir_2024
            )


            # =================================================
            # PIXELS VALIDES COMMUNS
            # =================================================

            valid = (
                valid_2020 &
                valid_2024 &
                (red_2020 > 0) &
                (nir_2020 > 0) &
                (red_2024 > 0) &
                (nir_2024 > 0)
            )


            # =================================================
            # DIFFÉRENCE NDVI
            # =================================================

            ndvi_difference = (
                ndvi_2024 - ndvi_2020
            )


            # =================================================
            # CRÉATION DU MASQUE
            # =================================================

            mask = np.zeros(
                (tile_height, tile_width),
                dtype=np.uint8
            )

            # 0 = pas de déforestation détectée
            # 1 = diminution importante du NDVI

            mask[
                valid &
                (ndvi_difference < THRESHOLD)
            ] = 1


            # =================================================
            # STATISTIQUES
            # =================================================

            valid_pixels = np.sum(valid)

            deforestation_pixels = np.sum(
                mask == 1
            )

            total_valid += valid_pixels
            total_deforestation += (
                deforestation_pixels
            )


            # =================================================
            # TRANSFORMATION GÉOGRAPHIQUE
            # =================================================

            transform = rasterio.windows.transform(
                window,
                src_2020.transform
            )


            # =================================================
            # PROFIL DU FICHIER
            # =================================================

            profile = src_2020.profile.copy()

            profile.update(
                driver="GTiff",
                dtype=rasterio.uint8,
                count=1,
                width=tile_width,
                height=tile_height,
                transform=transform,
                compress="lzw",
                nodata=0
            )


            # =================================================
            # NOM DU MASQUE
            # =================================================

            output_file = os.path.join(
                OUTPUT_DIR,
                f"mask_x{x}_y{y}.tif"
            )


            # =================================================
            # SAUVEGARDE
            # =================================================

            with rasterio.open(
                output_file,
                "w",
                **profile
            ) as dst:

                dst.write(mask, 1)


            total_tiles += 1


            # Affichage de progression

            if total_tiles % 100 == 0:

                print(
                    f"Tuiles traitées : {total_tiles}"
                )


# ============================================================
# RÉSULTATS
# ============================================================

print()
print("======================================")
print("TRAITEMENT TERMINÉ")
print("======================================")

print(
    "Nombre de tuiles :",
    total_tiles
)

print(
    "Pixels valides :",
    total_valid
)

print(
    "Pixels classés déforestation :",
    total_deforestation
)


if total_valid > 0:

    percentage = (
        total_deforestation /
        total_valid *
        100
    )

    print(
        "Déforestation potentielle :",
        round(percentage, 2),
        "%"
    )


print()
print(
    "Masques sauvegardés dans :",
    OUTPUT_DIR
)
