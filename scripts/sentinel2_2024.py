import ee

# --------------------------------------------------
# 1. Connexion à Google Earth Engine
# --------------------------------------------------

ee.Initialize(project="deforestation-ai-projet")

# --------------------------------------------------
# 2. Zone d'étude : Manaus, Amazonas, Brésil
# --------------------------------------------------

roi = ee.Geometry.Rectangle([
    -60.30, -3.35,
    -59.75, -2.85
])

# --------------------------------------------------
# 3. Recherche des images Sentinel-2
# --------------------------------------------------

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(roi)
    .filterDate("2024-06-01", "2024-09-30")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .sort("CLOUDY_PIXEL_PERCENTAGE")
)

# --------------------------------------------------
# 4. Sélection de la meilleure image
# --------------------------------------------------

nombre_images = collection.size().getInfo()

print("Nombre d'images :", nombre_images)

if nombre_images == 0:
    print("Aucune image trouvée.")
    exit()

image = ee.Image(collection.first())

image_id = image.get("system:index").getInfo()
nuages = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

print("Image sélectionnée :", image_id)
print("Couverture nuageuse :", nuages, "%")

# --------------------------------------------------
# 5. Sélection des bandes utiles
# --------------------------------------------------

image = image.select([
    "B2",
    "B3",
    "B4",
    "B8"
])

# --------------------------------------------------
# 6. Export vers Google Cloud Storage
# --------------------------------------------------

task = ee.batch.Export.image.toCloudStorage(
    image=image,
    description="sentinel2_manaus_2024",
    bucket="deforestation-ai-nadabh-2026",
    fileNamePrefix="raw/2024/sentinel2_manaus_2024",
    region=roi,
    scale=10,
    maxPixels=1e10,
    fileFormat="GeoTIFF"
)

task.start()

print("Export lancé.")
print("ID de la tâche :", task.id)
