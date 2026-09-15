import os
import sys
from datetime import datetime, timedelta

import ee
import numpy as np
import requests
import rasterio
from affine import Affine
from rasterio.enums import Resampling


PROJECT_ID = "deforestation-ai-projet"

OUTPUT_DIR = "v2/data/raw"

BANDS = [
    "B2",
    "B3",
    "B4",
    "B8",
]

REFERENCE_START = "2020-01-01"
REFERENCE_END = "2021-01-01"

IMAGE_SIZE = 256

CRS = "EPSG:32720"

MAX_CLOUD = 20


def initialize_earth_engine():
    """
    Initialise Google Earth Engine avec le projet GCP.
    """
    try:
        ee.Initialize(
            project=PROJECT_ID
        )
    except Exception:
        ee.Authenticate()
        ee.Initialize(
            project=PROJECT_ID
        )


def create_roi(
    latitude,
    longitude,
    size
):
    """
    Crée une zone carrée autour des coordonnées fournies.

    La taille est exprimée en mètres.
    """
    half_size = size / 2

    point = ee.Geometry.Point(
        [
            longitude,
            latitude
        ]
    )

    roi = point.buffer(
        half_size
    ).bounds()

    return roi


def get_collection(
    roi,
    start_date,
    end_date,
    max_cloud=20
):
    """
    Construit une collection Sentinel-2 filtrée
    par zone, période et couverture nuageuse.
    """
    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(
            roi
        )
        .filterDate(
            start_date,
            end_date
        )
        .filter(
            ee.Filter.lte(
                "CLOUDY_PIXEL_PERCENTAGE",
                max_cloud
            )
        )
        .select(
            BANDS
        )
    )

    return collection


def select_image(collection):
    """
    Sélectionne l'image présentant la plus faible
    couverture nuageuse.
    """
    size = collection.size().getInfo()

    if size == 0:
        return None

    sorted_collection = collection.sort(
        "CLOUDY_PIXEL_PERCENTAGE"
    )

    image = ee.Image(
        sorted_collection.first()
    )

    return image


def get_image_info(image):
    """
    Récupère la date et la couverture nuageuse
    d'une image Sentinel-2.
    """
    date = (
        ee.Date(
            image.get(
                "system:time_start"
            )
        )
        .format(
            "YYYY-MM-dd"
        )
        .getInfo()
    )

    cloud = image.get(
        "CLOUDY_PIXEL_PERCENTAGE"
    ).getInfo()

    return {
        "date": date,
        "cloud_percentage": cloud,
    }


def download_image(
    image,
    roi,
    output_path
):
    """
    Télécharge une image Sentinel-2 puis la normalise
    exactement en 256x256 pixels pour être compatible
    avec le modèle U-Net.
    """
    url = image.getDownloadURL(
        {
            "region": roi,
            "dimensions": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "crs": CRS,
            "fileFormat": "GeoTIFF",
            "format": "GEO_TIFF",
        }
    )

    response = requests.get(
        url,
        timeout=120
    )

    response.raise_for_status()

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    temporary_path = (
        output_path
        + ".tmp.tif"
    )

    with open(
        temporary_path,
        "wb"
    ) as file:
        file.write(
            response.content
        )

    with rasterio.open(
        temporary_path
    ) as src:

        if src.count != 4:
            raise RuntimeError(
                f"Nombre de bandes incorrect : "
                f"{src.count}. Attendu : 4."
            )

        data = src.read(
            out_shape=(
                src.count,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            resampling=Resampling.bilinear
        )

        transform = (
            src.transform
            * Affine.scale(
                src.width / IMAGE_SIZE,
                src.height / IMAGE_SIZE
            )
        )

        profile = src.profile.copy()

        profile.update(
            {
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE,
                "transform": transform,
                "count": 4,
                "dtype": data.dtype,
            }
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:
            dst.write(
                data
            )

    if os.path.exists(
        temporary_path
    ):
        os.remove(
            temporary_path
        )

    with rasterio.open(
        output_path
    ) as check:

        if (
            check.width != IMAGE_SIZE
            or check.height != IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Image finale incorrecte : "
                f"{check.width}x{check.height}. "
                f"Attendu : "
                f"{IMAGE_SIZE}x{IMAGE_SIZE}."
            )

    print(
        f"Taille finale : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )


def download_reference(
    roi,
    max_cloud=20
):
    """
    Télécharge l'image Sentinel-2 de référence
    pour l'année 2020.
    """
    collection = get_collection(
        roi,
        REFERENCE_START,
        REFERENCE_END,
        max_cloud=max_cloud
    )

    image = select_image(
        collection
    )

    if image is None:
        raise RuntimeError(
            "Aucune image Sentinel-2 disponible "
            "pour 2020 avec une couverture "
            f"nuageuse <= {max_cloud}%."
        )

    info = get_image_info(
        image
    )

    print(
        f"Référence : "
        f"{info['date']} "
        f"({info['cloud_percentage']:.2f}% nuages)"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "sentinel2_reference.tif"
    )

    download_image(
        image,
        roi,
        output_path
    )

    return {
        "path": output_path,
        "date": info["date"],
        "cloud_percentage": info[
            "cloud_percentage"
        ],
    }


def download_recent(
    roi,
    days=90,
    max_cloud=20
):
    """
    Télécharge l'image Sentinel-2 la plus récente
    dans la fenêtre temporelle demandée.
    """
    end_date = datetime.utcnow()

    start_date = (
        end_date
        - timedelta(
            days=days
        )
    )

    collection = get_collection(
        roi,
        start_date.strftime(
            "%Y-%m-%d"
        ),
        end_date.strftime(
            "%Y-%m-%d"
        ),
        max_cloud=max_cloud
    )

    image = select_image(
        collection
    )

    if image is None:
        raise RuntimeError(
            "Aucune image Sentinel-2 récente "
            "disponible avec une couverture "
            f"nuageuse <= {max_cloud}%."
        )

    info = get_image_info(
        image
    )

    print(
        f"Récente : "
        f"{info['date']} "
        f"({info['cloud_percentage']:.2f}% nuages)"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "sentinel2_recent.tif"
    )

    download_image(
        image,
        roi,
        output_path
    )

    return {
        "path": output_path,
        "date": info["date"],
        "cloud_percentage": info[
            "cloud_percentage"
        ],
    }


def download_sentinel_images(
    latitude,
    longitude,
    size,
    recent_days=90,
    max_cloud=20
):
    """
    Télécharge automatiquement les deux images
    Sentinel-2 nécessaires au modèle.

    Une image de référence est sélectionnée en 2020.
    Une image récente est sélectionnée dans la fenêtre
    temporelle demandée.
    """
    if not -90 <= latitude <= 90:
        raise ValueError(
            "Latitude invalide."
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude invalide."
        )

    if size <= 0:
        raise ValueError(
            "La taille de la zone doit être positive."
        )

    if recent_days <= 0:
        raise ValueError(
            "Le nombre de jours doit être positif."
        )

    if not 0 <= max_cloud <= 100:
        raise ValueError(
            "Le pourcentage de nuages doit être "
            "compris entre 0 et 100."
        )

    initialize_earth_engine()

    roi = create_roi(
        latitude,
        longitude,
        size
    )

    print()
    print(
        f"REFERENCE 2020 - "
        f"couverture nuageuse maximale : "
        f"{max_cloud}%"
    )

    reference = download_reference(
        roi,
        max_cloud=max_cloud
    )

    print()
    print(
        f"RECENTE - fenêtre : "
        f"{recent_days} jours - "
        f"couverture nuageuse maximale : "
        f"{max_cloud}%"
    )

    recent = download_recent(
        roi,
        days=recent_days,
        max_cloud=max_cloud
    )

    return {
        "reference": reference,
        "recent": recent,
    }


if __name__ == "__main__":
    print(
        "Module Earth Engine V2 chargé."
    )
