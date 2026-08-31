import rasterio
from rasterio.windows import Window
import numpy as np
import os


# ==================================================
# PARAMÈTRES
# ==================================================

TILE_SIZE = 256

IMAGES = {
    2020: "data/raw/2020/sentinel2_manaus_2020.tif",
    2024: "data/raw/2024/sentinel2_manaus_2024.tif"
}


# ==================================================
# FONCTION DE CRÉATION DES TUILES
# ==================================================

def create_tiles(input_path, output_dir, year):

    os.makedirs(output_dir, exist_ok=True)

    print("\n====================================")
    print("Année :", year)
    print("====================================")

    with rasterio.open(input_path) as src:

        width = src.width
        height = src.height

        print("Largeur :", width)
        print("Hauteur :", height)

        tile_count = 0
        skipped_count = 0

        # ------------------------------------------
        # Parcours de l'image
        # ------------------------------------------

        for y in range(0, height, TILE_SIZE):

            for x in range(0, width, TILE_SIZE):

                # ----------------------------------
                # Taille réelle de la fenêtre
                # ----------------------------------

                w = min(
                    TILE_SIZE,
                    width - x
                )

                h = min(
                    TILE_SIZE,
                    height - y
                )

                # ----------------------------------
                # On ignore les tuiles incomplètes
                # sur les bords
                # ----------------------------------

                if w != TILE_SIZE or h != TILE_SIZE:

                    skipped_count += 1
                    continue

                window = Window(
                    x,
                    y,
                    TILE_SIZE,
                    TILE_SIZE
                )

                # ----------------------------------
                # Lecture des 4 bandes
                # ----------------------------------

                data = src.read(
                    window=window
                )

                # ----------------------------------
                # Vérification des pixels valides
                # ----------------------------------

                valid_mask = np.all(
                    data > 0,
                    axis=0
                )

                valid_ratio = (
                    np.sum(valid_mask)
                    /
                    valid_mask.size
                )

                # ----------------------------------
                # On ignore les tuiles contenant
                # trop de pixels invalides
                # ----------------------------------

                if valid_ratio < 0.80:

                    skipped_count += 1
                    continue

                # ----------------------------------
                # Nom du fichier
                # ----------------------------------

                filename = (
                    f"manaus_{year}_"
                    f"x{x}_y{y}.tif"
                )

                output_path = os.path.join(
                    output_dir,
                    filename
                )

                # ----------------------------------
                # Métadonnées
                # ----------------------------------

                profile = src.profile.copy()

                profile.update({
                    "height": TILE_SIZE,
                    "width": TILE_SIZE,
                    "transform": src.window_transform(
                        window
                    )
                })

                # ----------------------------------
                # Sauvegarde
                # ----------------------------------

                with rasterio.open(
                    output_path,
                    "w",
                    **profile
                ) as dst:

                    dst.write(data)

                tile_count += 1

        print("Tuiles créées :", tile_count)
        print("Tuiles ignorées :", skipped_count)


# ==================================================
# 2020
# ==================================================

create_tiles(
    IMAGES[2020],
    "data/processed/images/2020",
    2020
)


# ==================================================
# 2024
# ==================================================

create_tiles(
    IMAGES[2024],
    "data/processed/images/2024",
    2024
)


print("\n====================================")
print("CRÉATION DES TUILES TERMINÉE")
print("====================================")
