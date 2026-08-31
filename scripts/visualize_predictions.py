import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from scripts.dataset import DeforestationDataset
from scripts.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "data/ml_dataset/test"
MODEL_PATH = "models/best_model.pth"

OUTPUT_DIR = "reports/predictions"

BATCH_SIZE = 1
NUMBER_OF_IMAGES = 5

DEVICE = torch.device("cpu")


# ============================================================
# DOSSIER DE SORTIE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

dataset = DeforestationDataset(TEST_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODELE
# ============================================================

model = UNet(
    in_channels=8,
    out_channels=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


print("=" * 60)
print("VISUALISATION DES PREDICTIONS")
print("=" * 60)

print("Modèle :", MODEL_PATH)
print("Epoch :", checkpoint["epoch"])
print("Images à visualiser :", NUMBER_OF_IMAGES)


# ============================================================
# PREDICTIONS
# ============================================================

with torch.no_grad():

    for index, (images, masks) in enumerate(loader):

        if index >= NUMBER_OF_IMAGES:
            break

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        logits = model(images)

        probabilities = torch.sigmoid(logits)

        THRESHOLD = 0.70

        predictions = (
            probabilities >= THRESHOLD
        ).float()

        # ----------------------------------------------------
        # Conversion NumPy
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Conversion NumPy
        # ----------------------------------------------------

        image = images[0].cpu().numpy()

        mask = masks[0].cpu().numpy()

        prediction = (
            predictions[0, 0]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Création des images RGB
        # ----------------------------------------------------

        # Image 2020
        rgb_2020 = np.stack(
            [
                image[2],
                image[1],
                image[0]
            ],
            axis=-1
        )

        # Image 2024
        rgb_2024 = np.stack(
            [
                image[6],
                image[5],
                image[4]
            ],
            axis=-1
        )


        # ----------------------------------------------------
        # Normalisation pour affichage
        # ----------------------------------------------------

        def normalize_rgb(img):

            img = np.clip(
                img,
                0,
                1
            )

            min_value = img.min()
            max_value = img.max()

            if max_value > min_value:

                img = (
                    img - min_value
                ) / (
                    max_value - min_value
                )

            return img


        rgb_2020 = normalize_rgb(
            rgb_2020
        )

        rgb_2024 = normalize_rgb(
            rgb_2024
        )


        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 10)
        )


        axes[0, 0].imshow(
            rgb_2020
        )

        axes[0, 0].set_title(
            "Image Sentinel-2 — 2020"
        )

        axes[0, 0].axis("off")


        axes[0, 1].imshow(
            rgb_2024
        )

        axes[0, 1].set_title(
            "Image Sentinel-2 — 2024"
        )

        axes[0, 1].axis("off")


        axes[1, 0].imshow(
            mask,
            vmin=0,
            vmax=1
        )

        axes[1, 0].set_title(
            "Masque réel"
        )

        axes[1, 0].axis("off")


        axes[1, 1].imshow(
            prediction,
            vmin=0,
            vmax=1
        )

        axes[1, 1].set_title(
            "Prédiction U-Net — seuil 0.7"
        )

        axes[1, 1].axis("off")


        plt.tight_layout()


        # ----------------------------------------------------
        # SAUVEGARDE
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"prediction_{index + 1}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()


        print(
            f"Image {index + 1} sauvegardée : "
            f"{output_path}"
        )


print()
print("=" * 60)
print("VISUALISATION TERMINEE")
print("=" * 60)
print(f"Les images sont dans : {OUTPUT_DIR}")
