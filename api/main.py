import io
import os
import sys

import numpy as np
import rasterio
import torch

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ============================================================
# CHEMIN DU PROJET
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORTS PROJET
# ============================================================

from scripts.model import UNet
from v2.scripts.pipeline_v2 import run_v2


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
    version="2.0.0"
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
# MODELE DE REQUETE V2
# ============================================================

class AnalyzeV2Request(BaseModel):

    latitude: float
    longitude: float

    size: int = 2560

    recent_days: int = 90


# ============================================================
# ENDPOINT RACINE
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Deforestation AI API",
        "status": "online",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/analyze",
            "/compare",
            "/analyze-v2"
        ]
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
# ANALYSE V1
# ============================================================

@app.post("/analyze")
async def analyze(
    image_2020: UploadFile = File(...),
    image_2024: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Lecture
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
    # Combinaison
    # --------------------------------------------------------

    image = np.concatenate(
        [
            image_2020_array,
            image_2024_array
        ],
        axis=0
    )

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
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
    # Masque
    # --------------------------------------------------------

    mask = predictions[
        0, 0
    ].cpu().numpy().astype(np.uint8)

    # --------------------------------------------------------
    # Calcul
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
    # Résultat
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
# COMPARAISON V1 : 2020 → 2024
# ============================================================

@app.post("/compare")
async def compare(
    image_2020: UploadFile = File(...),
    image_2024: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Lecture
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
    # Combinaison
    # --------------------------------------------------------

    image = np.concatenate(
        [
            image_2020_array,
            image_2024_array
        ],
        axis=0
    )

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(image_tensor)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= THRESHOLD
        ).float()

    # --------------------------------------------------------
    # Masque
    # --------------------------------------------------------

    mask = predictions[
        0, 0
    ].cpu().numpy().astype(np.uint8)

    # --------------------------------------------------------
    # Calcul
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
    # Résultat
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


# ============================================================
# ANALYSE V2
# ============================================================

@app.post("/analyze-v2")
def analyze_v2(request: AnalyzeV2Request):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not -90 <= request.latitude <= 90:

        raise HTTPException(
            status_code=400,
            detail="Latitude invalide."
        )

    if not -180 <= request.longitude <= 180:

        raise HTTPException(
            status_code=400,
            detail="Longitude invalide."
        )

    if request.size <= 0:

        raise HTTPException(
            status_code=400,
            detail="La taille de la zone doit être positive."
        )

    if request.recent_days <= 0:

        raise HTTPException(
            status_code=400,
            detail="recent_days doit être supérieur à 0."
        )

    # --------------------------------------------------------
    # Pipeline V2
    # --------------------------------------------------------

    try:

        result = run_v2(
            latitude=request.latitude,
            longitude=request.longitude,
            size=request.size,
            recent_days=request.recent_days
        )

    except Exception as e:

        print(
            "ERREUR PIPELINE V2 :",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse V2 : {e}"
        )

    # --------------------------------------------------------
    # URLs des résultats
    # --------------------------------------------------------

    job_id = result["job_id"]

    result["mask_url"] = (
        f"/v2/results/{job_id}/mask"
    )

    result["probability_url"] = (
        f"/v2/results/{job_id}/probability"
    )

    return result


# ============================================================
# RESULTAT V2 : MASQUE
# ============================================================

@app.get("/v2/results/{job_id}/mask")
def get_v2_mask(job_id: str):

    path = os.path.join(
        "v2",
        "reports",
        "results",
        job_id,
        "mask.png"
    )

    if not os.path.isfile(path):

        raise HTTPException(
            status_code=404,
            detail="Masque introuvable."
        )

    return FileResponse(
        path,
        media_type="image/png"
    )


# ============================================================
# RESULTAT V2 : PROBABILITES
# ============================================================

@app.get("/v2/results/{job_id}/probability")
def get_v2_probability(job_id: str):

    path = os.path.join(
        "v2",
        "reports",
        "results",
        job_id,
        "probability.png"
    )

    if not os.path.isfile(path):

        raise HTTPException(
            status_code=404,
            detail="Carte de probabilité introuvable."
        )

    return FileResponse(
        path,
        media_type="image/png"
    )
