import argparse
import os
from datetime import datetime, timedelta, timezone
import urllib.request

import ee
import rasterio
from rasterio.enums import Resampling
from affine import Affine


PROJECT_ID = "deforestation-ai-projet"
OUTPUT_DIR = "v2/data/raw"

BANDS = ["B2", "B3", "B4", "B8"]

REFERENCE_START = "2020-01-01"
REFERENCE_END = "2021-01-01"

IMAGE_SIZE = 256
CRS = "EPSG:32720"


def initialize_earth_engine():
    ee.Initialize(project=PROJECT_ID)
    print("Earth Engine initialisé")


def create_roi(latitude, longitude, size):
    point = ee.Geometry.Point([
        longitude,
        latitude
    ])

    roi = point.buffer(size / 2).bounds()

    return roi


def get_image(
    roi,
    start_date,
    end_date,
    max_cloud,
    description
):

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                max_cloud
            )
        )
        .select(BANDS)
        .sort(
            "system:time_start",
            False
        )
    )

    count = collection.size().getInfo()

    print(
        f"{description} - images trouvées : {count}"
    )

    if count == 0:
        raise RuntimeError(
            f"Aucune image Sentinel-2 trouvée pour "
            f"{description}."
        )

    image = ee.Image(
        collection.first()
    )

    date = (
        image
        .date()
        .format("YYYY-MM-dd")
        .getInfo()
    )

    cloud = image.get(
        "CLOUDY_PIXEL_PERCENTAGE"
    ).getInfo()

    print(
        f"{description} - image sélectionnée : {date}"
    )

    print(
        f"{description} - couverture nuageuse : "
        f"{cloud:.2f}%"
    )

    return image, date, cloud


def download_image(
    image,
    roi,
    output_path,
    name
):

    image = (
        image
        .select(BANDS)
        .clip(roi)
    )

    url = image.getDownloadURL({
        "name": name,
        "bands": BANDS,
        "region": roi,
        "dimensions": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
        "crs": CRS,
        "filePerBand": False,
        "format": "GEO_TIFF"
    })

    print(
        f"\nURL de téléchargement générée pour {name}"
    )

    temporary_path = output_path + ".tmp.tif"

    urllib.request.urlretrieve(
        url,
        temporary_path
    )

    print(
        f"Image Earth Engine téléchargée : "
        f"{temporary_path}"
    )

    with rasterio.open(
        temporary_path
    ) as src:

        if src.count != len(BANDS):
            raise RuntimeError(
                f"Nombre de bandes incorrect : "
                f"{src.count}. Attendu : {len(BANDS)}."
            )

        print(
            f"Taille reçue : "
            f"{src.width}x{src.height}"
        )

        data = src.read(
            out_shape=(
                src.count,
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            resampling=Resampling.bilinear
        )

        transform = src.transform * Affine.scale(
            src.width / IMAGE_SIZE,
            src.height / IMAGE_SIZE
        )

        profile = src.profile.copy()

        profile.update({
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "count": len(BANDS),
            "transform": transform,
            "crs": CRS,
            "dtype": data.dtype
        })

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(data)

    os.remove(temporary_path)

    with rasterio.open(
        output_path
    ) as check:

        print(
            f"Image finale : "
            f"{check.width}x{check.height}"
        )

        if (
            check.width != IMAGE_SIZE
            or check.height != IMAGE_SIZE
        ):
            raise RuntimeError(
                f"Erreur : image finale "
                f"{check.width}x{check.height}. "
                f"Attendu : "
                f"{IMAGE_SIZE}x{IMAGE_SIZE}."
            )

    print(
        f"Image normalisée : "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Téléchargement automatique de deux "
            "images Sentinel-2 pour une même zone"
        )
    )

    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude du centre"
    )

    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude du centre"
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
        help="Nombre de jours pour rechercher "
             "l'image récente"
    )

    parser.add_argument(
        "--max-cloud",
        type=float,
        default=20,
        help="Couverture nuageuse maximale en %"
    )

    args = parser.parse_args()

    if not -90 <= args.lat <= 90:
        raise ValueError("Latitude invalide.")

    if not -180 <= args.lon <= 180:
        raise ValueError("Longitude invalide.")

    if args.size <= 0:
        raise ValueError(
            "La taille de la zone doit être positive."
        )

    if args.days <= 0:
        raise ValueError(
            "Le nombre de jours doit être positif."
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    reference_path = os.path.join(
        OUTPUT_DIR,
        "sentinel2_reference.tif"
    )

    recent_path = os.path.join(
        OUTPUT_DIR,
        "sentinel2_recent.tif"
    )

    print("=" * 60)
    print("TELECHARGEMENT SENTINEL-2 - V2")
    print("=" * 60)

    print(
        f"Latitude       : {args.lat}"
    )

    print(
        f"Longitude      : {args.lon}"
    )

    print(
        f"Taille zone    : {args.size} m"
    )

    print(
        f"Image récente  : derniers {args.days} jours"
    )

    print(
        f"Nuages max     : {args.max_cloud}%"
    )

    print(
        f"Bandes         : {', '.join(BANDS)}"
    )

    print(
        f"Dimensions     : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print()

    initialize_earth_engine()

    roi = create_roi(
        args.lat,
        args.lon,
        args.size
    )

    print()
    print("-" * 60)
    print("RECHERCHE IMAGE DE REFERENCE")
    print("-" * 60)

    reference_image, reference_date, reference_cloud = (
        get_image(
            roi,
            REFERENCE_START,
            REFERENCE_END,
            args.max_cloud,
            "REFERENCE 2020"
        )
    )

    print()
    print("-" * 60)
    print("RECHERCHE IMAGE RECENTE")
    print("-" * 60)

    end_datetime = datetime.now(
        timezone.utc
    )

    start_datetime = (
        end_datetime
        - timedelta(days=args.days)
    )

    recent_start = (
        start_datetime
        .strftime("%Y-%m-%d")
    )

    recent_end = (
        end_datetime
        .strftime("%Y-%m-%d")
    )

    recent_image, recent_date, recent_cloud = (
        get_image(
            roi,
            recent_start,
            recent_end,
            args.max_cloud,
            "RECENTE"
        )
    )

    print()
    print("-" * 60)
    print("TELECHARGEMENT IMAGE DE REFERENCE")
    print("-" * 60)

    download_image(
        reference_image,
        roi,
        reference_path,
        "sentinel2_reference"
    )

    print()
    print("-" * 60)
    print("TELECHARGEMENT IMAGE RECENTE")
    print("-" * 60)

    download_image(
        recent_image,
        roi,
        recent_path,
        "sentinel2_recent"
    )

    print()
    print("=" * 60)
    print("TELECHARGEMENT TERMINE")
    print("=" * 60)

    print(
        f"Reference : {reference_date} "
        f"({reference_cloud:.2f}% nuages)"
    )

    print(
        f"Recente   : {recent_date} "
        f"({recent_cloud:.2f}% nuages)"
    )

    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Recente   : {recent_path}"
    )

    print(
        f"Dimensions : {IMAGE_SIZE}x{IMAGE_SIZE}"
    )


if __name__ == "__main__":
    main()
