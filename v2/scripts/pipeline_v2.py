import os
import argparse
import sys
import uuid

import numpy as np
import rasterio
import torch
from PIL import Image


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORTS DU PROJET
# ============================================================

from scripts.model import UNet
from v2.scripts.earth_engine_v2 import download_sentinel_images


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_model.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "v2",
    "reports",
    "results"
)

THRESHOLD = 0.70
EXPECTED_BANDS = 4
EXPECTED_SIZE = 256


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

def load_model():
    device = torch.device("cpu")

    model = UNet(
        in_channels=8,
        out_channels=1
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# LECTURE D'UNE IMAGE SENTINEL-2
# ============================================================

def read_sentinel_image(path):
    with rasterio.open(path) as src:

        image = src.read()

        profile = src.profile.copy()

        width = src.width
        height = src.height
        count = src.count
        crs = src.crs
        bounds = src.bounds

    if count != EXPECTED_BANDS:
        raise ValueError(
            f"Nombre de bandes incorrect : "
            f"{count}. Attendu : {EXPECTED_BANDS}."
        )

    if height != EXPECTED_SIZE or width != EXPECTED_SIZE:
        raise ValueError(
            f"Taille incorrecte : "
            f"{width}x{height}. "
            f"Attendu : {EXPECTED_SIZE}x{EXPECTED_SIZE}."
        )

    image = image.astype(np.float32)

    # Même normalisation que pendant l'entraînement V1
    image = image / 10000.0

    # Sécurité
    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image, profile, crs, bounds


# ============================================================
# PREPARATION DU TENSOR
# ============================================================

def prepare_input(reference_path, recent_path):

    reference, ref_profile, ref_crs, ref_bounds = (
        read_sentinel_image(reference_path)
    )

    recent, recent_profile, recent_crs, recent_bounds = (
        read_sentinel_image(recent_path)
    )

    # Vérification géométrique
    if reference.shape != recent.shape:
        raise ValueError(
            "Les deux images ont des dimensions différentes."
        )

    if ref_crs != recent_crs:
        raise ValueError(
            "Les deux images ont des CRS différents."
        )

    if not np.allclose(
        [
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top
        ],
        [
            recent_bounds.left,
            recent_bounds.bottom,
            recent_bounds.right,
            recent_bounds.top
        ],
        rtol=0,
        atol=1e-6
    ):
        raise ValueError(
            "Les deux images n'ont pas la même emprise."
        )

    # 4 bandes 2020 + 4 bandes récentes
    combined = np.concatenate(
        [
            reference,
            recent
        ],
        axis=0
    )

    if combined.shape != (
        8,
        EXPECTED_SIZE,
        EXPECTED_SIZE
    ):
        raise ValueError(
            f"Shape inattendue : {combined.shape}"
        )

    tensor = torch.from_numpy(
        combined
    ).unsqueeze(0)

    return tensor, ref_profile


# ============================================================
# SAUVEGARDE DES RESULTATS
# ============================================================

def save_results(
    probability,
    mask,
    job_id
):
    job_dir = os.path.join(
        OUTPUT_DIR,
        job_id
    )

    os.makedirs(
        job_dir,
        exist_ok=True
    )

    mask_path = os.path.join(
        job_dir,
        "mask.png"
    )

    probability_path = os.path.join(
        job_dir,
        "probability.png"
    )

    # Masque binaire : 0 / 255
    mask_image = (
        mask.astype(np.uint8) * 255
    )

    Image.fromarray(
        mask_image,
        mode="L"
    ).save(mask_path)

    # Probabilité : 0-1 -> 0-255
    probability_image = (
        probability * 255
    ).clip(
        0,
        255
    ).astype(np.uint8)

    Image.fromarray(
        probability_image,
        mode="L"
    ).save(probability_path)

    return {
        "mask_path": mask_path,
        "probability_path": probability_path
    }


# ============================================================
# PIPELINE COMPLET V2
# ============================================================

def run_v2(
    latitude,
    longitude,
    size=2560,
    recent_days=90,
    max_cloud=20
):

    # --------------------------------------------------------
    # IDENTIFIANT UNIQUE
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex

    print("=" * 60)
    print("DEFORESTATION AI - PIPELINE V2")
    print("=" * 60)

    print(f"Latitude       : {latitude}")
    print(f"Longitude      : {longitude}")
    print(f"Taille zone    : {size} m")
    print(f"Fenêtre récente: {recent_days} jours")
    print(f"Nuages max     : {max_cloud}%")
    print(f"Job ID         : {job_id}")

    # --------------------------------------------------------
    # 1. EARTH ENGINE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("1. TELECHARGEMENT SENTINEL-2")
    print("=" * 60)

    images = download_sentinel_images(
        latitude=latitude,
        longitude=longitude,
        size=size,
        recent_days=recent_days,
        max_cloud=max_cloud
    )

    reference = images["reference"]
    recent = images["recent"]

    reference_path = os.path.join(
        PROJECT_ROOT,
        reference["path"]
    )

    recent_path = os.path.join(
        PROJECT_ROOT,
        recent["path"]
    )

    print(
        f"Référence : {reference['date']} "
        f"({reference['cloud_percentage']:.2f}% nuages)"
    )

    print(
        f"Récente    : {recent['date']} "
        f"({recent['cloud_percentage']:.2f}% nuages)"
    )

    # --------------------------------------------------------
    # 2. PREPROCESSING
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. PRETRAITEMENT")
    print("=" * 60)

    tensor, profile = prepare_input(
        reference_path,
        recent_path
    )

    print(
        f"Tensor : {tuple(tensor.shape)}"
    )

    print(
        f"Min    : {tensor.min().item():.6f}"
    )

    print(
        f"Max    : {tensor.max().item():.6f}"
    )

    # --------------------------------------------------------
    # 3. MODELE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("3. INFERENCE UNET")
    print("=" * 60)

    model, device = load_model()

    print(f"Device : {device}")

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Paramètres : {total_parameters:,}"
    )

    # --------------------------------------------------------
    # 4. PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            tensor.to(device)
        )

        probability = torch.sigmoid(
            output
        )

    probability = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # 5. MASQUE
    # --------------------------------------------------------

    mask = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    total_pixels = mask.size

    deforested_pixels = int(
        mask.sum()
    )

    deforestation_percentage = (
        deforested_pixels
        / total_pixels
        * 100
    )

    print(
        f"Probabilité min : "
        f"{probability.min():.6f}"
    )

    print(
        f"Probabilité max : "
        f"{probability.max():.6f}"
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
        f"{deforestation_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # 6. SAUVEGARDE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("4. SAUVEGARDE DES RESULTATS")
    print("=" * 60)

    result_files = save_results(
        probability,
        mask,
        job_id
    )

    print(
        f"Masque : {result_files['mask_path']}"
    )

    print(
        f"Probabilités : "
        f"{result_files['probability_path']}"
    )

    # --------------------------------------------------------
    # 7. RESULTAT
    # --------------------------------------------------------

    return {
        "job_id": job_id,

        "latitude": latitude,
        "longitude": longitude,
        "size_meters": size,

        "reference_date": reference["date"],
        "reference_cloud_percentage": (
            reference["cloud_percentage"]
        ),

        "recent_date": recent["date"],
        "recent_cloud_percentage": (
            recent["cloud_percentage"]
        ),

        "threshold": THRESHOLD,

        "total_pixels": total_pixels,
        "deforested_pixels": deforested_pixels,

        "deforestation_percentage": (
            deforestation_percentage
        ),

        "mask_path": result_files["mask_path"],
        "probability_path": (
            result_files["probability_path"]
        )
    }


# ============================================================
# EXECUTION EN LIGNE DE COMMANDE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Pipeline V2 de détection de déforestation"
    )

    parser.add_argument(
        "--lat",
        type=float,
        default=-3.4653,
        help="Latitude du centre de la zone"
    )

    parser.add_argument(
        "--lon",
        type=float,
        default=-62.2159,
        help="Longitude du centre de la zone"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=2560,
        help="Taille de la zone en mètres"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Nombre de jours pour rechercher l'image récente"
    )

    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20,
        help="Pourcentage maximal de couverture nuageuse"
    )

    args = parser.parse_args()

    result = run_v2(
        latitude=args.lat,
        longitude=args.lon,
        size=args.size,
        recent_days=args.days,
        max_cloud=args.max_cloud
    )

    print("\n" + "=" * 60)
    print("PIPELINE V2 TERMINE")
    print("=" * 60)

    print(
        f"Job ID : {result['job_id']}"
    )

    print(
        f"Déforestation : "
        f"{result['deforestation_percentage']:.2f}%"
    )
