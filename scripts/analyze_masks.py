import glob
import rasterio
import numpy as np


DATASET_DIR = "data/ml_dataset"


def analyze_split(split):

    masks_dir = f"{DATASET_DIR}/{split}/masks"

    files = glob.glob(
        f"{masks_dir}/*.tif"
    )

    total_pixels = 0
    deforestation_pixels = 0
    non_deforestation_pixels = 0

    for file in files:

        with rasterio.open(file) as src:
            mask = src.read(1)

        total_pixels += mask.size

        deforestation_pixels += np.sum(mask == 1)

        non_deforestation_pixels += np.sum(mask == 0)

    percentage_0 = (
        non_deforestation_pixels
        / total_pixels
        * 100
    )

    percentage_1 = (
        deforestation_pixels
        / total_pixels
        * 100
    )

    print("=" * 60)
    print(f"DATASET : {split.upper()}")
    print("=" * 60)

    print(f"Nombre de masques : {len(files)}")
    print(f"Pixels totaux : {total_pixels:,}")

    print(
        f"Classe 0 : "
        f"{non_deforestation_pixels:,} "
        f"({percentage_0:.2f} %)"
    )

    print(
        f"Classe 1 : "
        f"{deforestation_pixels:,} "
        f"({percentage_1:.2f} %)"
    )

    print()


# ============================================================
# ANALYSE DES TROIS DATASETS
# ============================================================

for split in [
    "train",
    "validation",
    "test"
]:

    analyze_split(split)
