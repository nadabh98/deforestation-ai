import os
import glob

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class DeforestationDataset(Dataset):
    """
    Dataset PyTorch pour la détection de déforestation.

    Chaque exemple contient :
        - une image Sentinel-2 de 2020 : 4 bandes
        - une image Sentinel-2 de 2024 : 4 bandes
        - un masque de déforestation : 0 ou 1

    L'entrée du modèle aura donc 8 canaux :
        B2_2020, B3_2020, B4_2020, B8_2020,
        B2_2024, B3_2024, B4_2024, B8_2024
    """

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.images_2020_dir = os.path.join(
            root_dir,
            "images_2020"
        )

        self.images_2024_dir = os.path.join(
            root_dir,
            "images_2024"
        )

        self.masks_dir = os.path.join(
            root_dir,
            "masks"
        )

        # --------------------------------------------------
        # Recherche des images 2020
        # --------------------------------------------------

        self.images_2020 = sorted(
            glob.glob(
                os.path.join(
                    self.images_2020_dir,
                    "*.tif"
                )
            )
        )

        # --------------------------------------------------
        # Vérification des fichiers correspondants
        # --------------------------------------------------

        self.samples = []

        for image_2020 in self.images_2020:

            filename = os.path.basename(image_2020)

            # Exemple :
            # manaus_2020_x768_y256.tif

            coordinates = filename.replace(
                "manaus_2020_",
                ""
            ).replace(
                ".tif",
                ""
            )

            image_2024 = os.path.join(
                self.images_2024_dir,
                f"manaus_2024_{coordinates}.tif"
            )

            mask = os.path.join(
                self.masks_dir,
                f"mask_{coordinates}.tif"
            )

            # On garde uniquement les couples complets

            if (
                os.path.exists(image_2024)
                and os.path.exists(mask)
            ):
                self.samples.append(
                    (
                        image_2020,
                        image_2024,
                        mask
                    )
                )

        print(
            f"Dataset chargé : {len(self.samples)} exemples"
        )

    def __len__(self):
        """
        Retourne le nombre total d'exemples.
        """

        return len(self.samples)

    def __getitem__(self, index):
        """
        Charge un exemple du dataset.
        """

        image_2020_path, image_2024_path, mask_path = (
            self.samples[index]
        )

        # --------------------------------------------------
        # Lecture image 2020
        # --------------------------------------------------

        with rasterio.open(image_2020_path) as src:

            image_2020 = src.read()

        # --------------------------------------------------
        # Lecture image 2024
        # --------------------------------------------------

        with rasterio.open(image_2024_path) as src:

            image_2024 = src.read()

        # --------------------------------------------------
        # Lecture du masque
        # --------------------------------------------------

        with rasterio.open(mask_path) as src:

            mask = src.read(1)

        # --------------------------------------------------
        # Conversion uint16 → float32
        # --------------------------------------------------

        image_2020 = image_2020.astype(
            np.float32
        )

        image_2024 = image_2024.astype(
            np.float32
        )

        # --------------------------------------------------
        # Normalisation Sentinel-2
        # --------------------------------------------------

        image_2020 = image_2020 / 10000.0

        image_2024 = image_2024 / 10000.0

        # --------------------------------------------------
        # Conversion du masque
        # --------------------------------------------------

        mask = mask.astype(
            np.float32
        )

        # --------------------------------------------------
        # Combinaison 2020 + 2024
        # --------------------------------------------------

        image = np.concatenate(
            [
                image_2020,
                image_2024
            ],
            axis=0
        )

        # --------------------------------------------------
        # Conversion en tenseurs PyTorch
        # --------------------------------------------------

        image = torch.from_numpy(
            image
        )

        mask = torch.from_numpy(
            mask
        )

        return image, mask
