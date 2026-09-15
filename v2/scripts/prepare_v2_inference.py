import os

import numpy as np
import rasterio
import torch


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_PATH = "v2/data/raw/sentinel2_reference.tif"
RECENT_PATH = "v2/data/raw/sentinel2_recent.tif"

OUTPUT_PATH = "v2/data/processed/inference_input.pt"

EXPECTED_BANDS = 4
EXPECTED_HEIGHT = 256
EXPECTED_WIDTH = 256

# Valeur maximale approximative des réflectances Sentinel-2 SR
SENTINEL_SCALE = 10000.0


# ============================================================
# CHARGEMENT D'UNE IMAGE
# ============================================================

def load_sentinel_image(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    with rasterio.open(path) as src:

        image = src.read()

        profile = src.profile

        print(f"\nFichier : {path}")
        print(f"Dimensions : {src.width} x {src.height}")
        print(f"Bandes : {src.count}")
        print(f"CRS : {src.crs}")
        print(f"Résolution : {src.res}")
        print(f"Type : {src.dtypes}")

    return image, profile


# ============================================================
# VERIFICATION DES IMAGES
# ============================================================

def verify_images(reference, recent):

    print("\n" + "=" * 60)
    print("VERIFICATION DES IMAGES")
    print("=" * 60)

    if reference.shape != recent.shape:

        raise ValueError(
            "Les deux images n'ont pas la même dimension.\n"
            f"Reference : {reference.shape}\n"
            f"Recent    : {recent.shape}"
        )

    if reference.shape[0] != EXPECTED_BANDS:

        raise ValueError(
            f"Nombre de bandes incorrect : "
            f"{reference.shape[0]}"
        )

    if reference.shape[1] != EXPECTED_HEIGHT:

        raise ValueError(
            f"Hauteur incorrecte : "
            f"{reference.shape[1]}"
        )

    if reference.shape[2] != EXPECTED_WIDTH:

        raise ValueError(
            f"Largeur incorrecte : "
            f"{reference.shape[2]}"
        )

    print("✓ Même nombre de bandes")
    print("✓ Même hauteur")
    print("✓ Même largeur")


# ============================================================
# NORMALISATION
# ============================================================

def normalize(image):

    image = image.astype(np.float32)

    image = image / SENTINEL_SCALE

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# CREATION DES 8 CANAUX
# ============================================================

def create_8_channels(reference, recent):

    reference = normalize(reference)
    recent = normalize(recent)

    # Ordre identique à celui utilisé pour entraîner le modèle V1 :
    #
    # B2_2020
    # B3_2020
    # B4_2020
    # B8_2020
    # B2_recent
    # B3_recent
    # B4_recent
    # B8_recent

    combined = np.concatenate(
        [
            reference,
            recent
        ],
        axis=0
    )

    return combined


# ============================================================
# CONVERSION PYTORCH
# ============================================================

def create_tensor(image):

    tensor = torch.from_numpy(image)

    # Ajout de la dimension batch
    #
    # Avant :
    # [8, 256, 256]
    #
    # Après :
    # [1, 8, 256, 256]

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PREPARATION INFERENCE V2")
    print("=" * 60)

    # --------------------------------------------------------
    # Chargement
    # --------------------------------------------------------

    reference, reference_profile = load_sentinel_image(
        REFERENCE_PATH
    )

    recent, recent_profile = load_sentinel_image(
        RECENT_PATH
    )

    # --------------------------------------------------------
    # Vérification
    # --------------------------------------------------------

    verify_images(
        reference,
        recent
    )

    # --------------------------------------------------------
    # Création des 8 canaux
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CREATION DES 8 CANAUX")
    print("=" * 60)

    combined = create_8_channels(
        reference,
        recent
    )

    print(
        "Shape avant PyTorch :",
        combined.shape
    )

    print(
        "Valeur minimale :",
        combined.min()
    )

    print(
        "Valeur maximale :",
        combined.max()
    )

    # --------------------------------------------------------
    # Tensor PyTorch
    # --------------------------------------------------------

    tensor = create_tensor(
        combined
    )

    print("\n" + "=" * 60)
    print("TENSOR PYTORCH")
    print("=" * 60)

    print(
        "Shape finale :",
        tuple(tensor.shape)
    )

    print(
        "Type :",
        tensor.dtype
    )

    print(
        "Min :",
        tensor.min().item()
    )

    print(
        "Max :",
        tensor.max().item()
    )

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    torch.save(
        tensor,
        OUTPUT_PATH
    )

    print("\n" + "=" * 60)
    print("PREPARATION TERMINEE")
    print("=" * 60)

    print(
        f"Tensor sauvegardé : {OUTPUT_PATH}"
    )

    print(
        "Entrée modèle attendue : [1, 8, 256, 256]"
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
