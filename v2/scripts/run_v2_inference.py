import os

import sys
import numpy as np
import torch
from PIL import Image


# Permet d'importer les scripts du projet principal
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)



from scripts.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = "v2/data/processed/inference_input.pt"
MODEL_PATH = "models/best_model.pth"

OUTPUT_DIR = "v2/reports"

MASK_PATH = os.path.join(
    OUTPUT_DIR,
    "v2_deforestation_mask.png"
)

PROBABILITY_PATH = os.path.join(
    OUTPUT_DIR,
    "v2_deforestation_probability.png"
)

THRESHOLD = 0.70

DEVICE = torch.device("cpu")


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

def load_model():

    print("=" * 60)
    print("CHARGEMENT DU MODELE")
    print("=" * 60)

    model = UNet()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Compatible avec plusieurs formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)

    model.eval()

    print(
        f"Modèle chargé : {MODEL_PATH}"
    )

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Paramètres : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    return model


# ============================================================
# CHARGEMENT DE L'ENTREE
# ============================================================

def load_input():

    print("\n" + "=" * 60)
    print("CHARGEMENT DE L'ENTREE V2")
    print("=" * 60)

    tensor = torch.load(
        INPUT_PATH,
        map_location=DEVICE
    )

    tensor = tensor.float()

    print(
        f"Fichier : {INPUT_PATH}"
    )

    print(
        f"Shape : {tuple(tensor.shape)}"
    )

    print(
        f"Type : {tensor.dtype}"
    )

    print(
        f"Min : {tensor.min().item():.6f}"
    )

    print(
        f"Max : {tensor.max().item():.6f}"
    )

    expected_shape = (
        1,
        8,
        256,
        256
    )

    if tuple(tensor.shape) != expected_shape:

        raise ValueError(
            "Shape incorrecte.\n"
            f"Attendu : {expected_shape}\n"
            f"Reçu    : {tuple(tensor.shape)}"
        )

    return tensor


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, tensor):

    print("\n" + "=" * 60)
    print("INFERENCE")
    print("=" * 60)

    with torch.no_grad():

        output = model(
            tensor
        )

    print(
        f"Sortie modèle : {tuple(output.shape)}"
    )

    # Le modèle produit des logits.
    # Conversion en probabilités avec sigmoid.

    probabilities = torch.sigmoid(
        output
    )

    print(
        f"Probabilité min : "
        f"{probabilities.min().item():.6f}"
    )

    print(
        f"Probabilité max : "
        f"{probabilities.max().item():.6f}"
    )

    return probabilities


# ============================================================
# CREATION DU MASQUE
# ============================================================

def create_mask(probabilities):

    print("\n" + "=" * 60)
    print("CREATION DU MASQUE")
    print("=" * 60)

    mask = (
        probabilities >= THRESHOLD
    )

    mask = mask.squeeze().cpu().numpy()

    mask = mask.astype(np.uint8)

    total_pixels = mask.size

    deforested_pixels = int(
        mask.sum()
    )

    percentage = (
        deforested_pixels
        / total_pixels
        * 100
    )

    print(
        f"Seuil : {THRESHOLD}"
    )

    print(
        f"Pixels totaux : {total_pixels:,}"
    )

    print(
        f"Pixels déforestés : "
        f"{deforested_pixels:,}"
    )

    print(
        f"Déforestation détectée : "
        f"{percentage:.2f}%"
    )

    return mask, percentage


# ============================================================
# SAUVEGARDE DU MASQUE
# ============================================================

def save_mask(mask):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # 0 = pas de déforestation
    # 1 = déforestation
    #
    # Pour l'image PNG :
    # 0 = noir
    # 255 = blanc

    mask_image = (
        mask * 255
    ).astype(np.uint8)

    Image.fromarray(
        mask_image
    ).save(
        MASK_PATH
    )

    print(
        f"\nMasque sauvegardé : {MASK_PATH}"
    )


# ============================================================
# SAUVEGARDE DES PROBABILITES
# ============================================================

def save_probability(probabilities):

    probability = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )

    probability_image = (
        probability * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(
        probability_image
    ).save(
        PROBABILITY_PATH
    )

    print(
        f"Carte de probabilité sauvegardée : "
        f"{PROBABILITY_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("DEFORESTATION AI - INFERENCE V2")
    print("=" * 60)

    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Données
    # --------------------------------------------------------

    tensor = load_input()

    # --------------------------------------------------------
    # Prédiction
    # --------------------------------------------------------

    probabilities = run_inference(
        model,
        tensor
    )

    # --------------------------------------------------------
    # Masque
    # --------------------------------------------------------

    mask, percentage = create_mask(
        probabilities
    )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    save_mask(
        mask
    )

    save_probability(
        probabilities
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INFERENCE V2 TERMINEE")
    print("=" * 60)

    print(
        f"Seuil utilisé : {THRESHOLD}"
    )

    print(
        f"Déforestation détectée : "
        f"{percentage:.2f}%"
    )

    print(
        f"Masque : {MASK_PATH}"
    )

    print(
        f"Probabilités : {PROBABILITY_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
