import io
import os

import numpy as np
import rasterio
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from scripts.model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_model.pth"

THRESHOLD = 0.70

DEVICE = torch.device("cpu")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Deforestation AI API",
    description=(
        "API de détection de la déforestation "
        "par imagerie satellite Sentinel-2"
    ),
    version="1.0.0"
)


# ============================================================
# CHARGEMENT DU MODELE
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
print("MODELE CHARGE")
print("=" * 60)
print("Modèle :", MODEL_PATH)
print("Epoch :", checkpoint["epoch"])
print("Threshold :", THRESHOLD)


# ============================================================
# ENDPOINT RACINE
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Deforestation AI API",
        "status": "online"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# LECTURE D'UNE IMAGE TIFF
# ============================================================

def read_tiff(file_bytes):

    try:

        with rasterio.open(
            io.BytesIO(file_bytes)
        ) as src:

            image = src.read()

            width = src.width
            height = src.height
            bands = src.count
            crs = str(src.crs)

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire le fichier TIFF : {e}"
        )

    return image, width, height, bands, crs


# ============================================================
# ANALYSE
# ============================================================

@app.post("/analyze")
async def analyze(
    image_2020: UploadFile = File(...),
    image_2024: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Lecture des fichiers
    # --------------------------------------------------------

    data_2020 = await image_2020.read()
    data_2024 = await image_2024.read()

    image_2020_array, width_2020, height_2020, bands_2020, crs_2020 = (
        read_tiff(data_2020)
    )

    image_2024_array, width_2024, height_2024, bands_2024, crs_2024 = (
        read_tiff(data_2024)
    )


    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if bands_2020 != 4:

        raise HTTPException(
            status_code=400,
            detail="L'image 2020 doit contenir exactement 4 bandes."
        )

    if bands_2024 != 4:

        raise HTTPException(
            status_code=400,
            detail="L'image 2024 doit contenir exactement 4 bandes."
        )

    if (
        width_2020 != width_2024
        or height_2020 != height_2024
    ):

        raise HTTPException(
            status_code=400,
            detail="Les images 2020 et 2024 doivent avoir la même taille."
        )


    # --------------------------------------------------------
    # Conversion float32
    # --------------------------------------------------------

    image_2020_array = image_2020_array.astype(
        np.float32
    )

    image_2024_array = image_2024_array.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalisation Sentinel-2
    # Même preprocessing que le dataset
    # --------------------------------------------------------

    image_2020_array /= 10000.0
    image_2024_array /= 10000.0


    # --------------------------------------------------------
    # Combinaison 2020 + 2024
    # --------------------------------------------------------

    image = np.concatenate(
        [
            image_2020_array,
            image_2024_array
        ],
        axis=0
    )


    # --------------------------------------------------------
    # NumPy → PyTorch
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= THRESHOLD
        ).float()


    # --------------------------------------------------------
    # MASQUE
    # --------------------------------------------------------

    mask = predictions[
        0, 0
    ].cpu().numpy().astype(
        np.uint8
    )


    # --------------------------------------------------------
    # CALCUL DU POURCENTAGE
    # --------------------------------------------------------

    total_pixels = mask.size

    deforested_pixels = int(
        mask.sum()
    )

    deforestation_percentage = (
        deforested_pixels
        / total_pixels
        * 100
    )


    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    return {
        "status": "success",
        "model_epoch": checkpoint["epoch"],
        "threshold": THRESHOLD,
        "width": width_2020,
        "height": height_2020,
        "crs_2020": crs_2020,
        "crs_2024": crs_2024,
        "total_pixels": total_pixels,
        "deforested_pixels": deforested_pixels,
        "deforestation_percentage": round(
            deforestation_percentage,
            2
        )
    }

# ============================================================
# COMPARAISON 2020 → 2024
# ============================================================

@app.post("/compare")
async def compare(
    image_2020: UploadFile = File(...),
    image_2024: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Lecture des fichiers
    # --------------------------------------------------------

    data_2020 = await image_2020.read()
    data_2024 = await image_2024.read()

    image_2020_array, width_2020, height_2020, bands_2020, crs_2020 = (
        read_tiff(data_2020)
    )

    image_2024_array, width_2024, height_2024, bands_2024, crs_2024 = (
        read_tiff(data_2024)
    )

    # --------------------------------------------------------
    # Vérifications
    # --------------------------------------------------------

    if bands_2020 != 4 or bands_2024 != 4:
        raise HTTPException(
            status_code=400,
            detail="Chaque image doit contenir exactement 4 bandes."
        )

    if (
        width_2020 != width_2024
        or height_2020 != height_2024
    ):
        raise HTTPException(
            status_code=400,
            detail="Les deux images doivent avoir la même taille."
        )

    # --------------------------------------------------------
    # Normalisation
    # --------------------------------------------------------

    image_2020_array = (
        image_2020_array.astype(np.float32)
        / 10000.0
    )

    image_2024_array = (
        image_2024_array.astype(np.float32)
        / 10000.0
    )

    # --------------------------------------------------------
    # Combinaison 2020 + 2024
    # --------------------------------------------------------

    image = np.concatenate(
        [
            image_2020_array,
            image_2024_array
        ],
        axis=0
    )

    # --------------------------------------------------------
    # Conversion PyTorch
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

    # --------------------------------------------------------
    # MASQUE
    # --------------------------------------------------------

    mask = predictions[
        0, 0
    ].cpu().numpy().astype(np.uint8)

    # --------------------------------------------------------
    # CALCUL
    # --------------------------------------------------------

    total_pixels = mask.size

    deforested_pixels = int(
        mask.sum()
    )

    deforestation_percentage = (
        deforested_pixels
        / total_pixels
        * 100
    )

    # --------------------------------------------------------
    # RESULTAT
    # --------------------------------------------------------

    return {
        "status": "success",
        "comparison": "2020 → 2024",
        "model_epoch": checkpoint["epoch"],
        "threshold": THRESHOLD,
        "image_2020": {
            "width": width_2020,
            "height": height_2020,
            "bands": bands_2020,
            "crs": crs_2020
        },
        "image_2024": {
            "width": width_2024,
            "height": height_2024,
            "bands": bands_2024,
            "crs": crs_2024
        },
        "total_pixels": total_pixels,
        "deforested_pixels": deforested_pixels,
        "deforestation_percentage": round(
            deforestation_percentage,
            2
        )
    }
