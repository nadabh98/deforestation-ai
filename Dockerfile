# Image Python légère pour exécuter l'API.
FROM python:3.12-slim

# Tous les fichiers de l'application seront placés dans /app.
WORKDIR /app

# Installation des bibliothèques système nécessaires à Rasterio/GDAL
# et à certaines dépendances scientifiques.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       g++ \
       libgdal-dev \
       libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier des dépendances Python.
COPY requirements.txt .

# Installation des dépendances Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code de l'API.
COPY api ./api

# Copie les scripts utilisés par le modèle V1.
COPY scripts ./scripts

# Copie le modèle entraîné.
COPY models ./models

# Copie l'ensemble du pipeline V2.
# Cette partie était absente de l'ancien Dockerfile.
COPY v2 ./v2

# Cloud Run fournit le port d'écoute via la variable PORT.
EXPOSE 8080

# Lancement de FastAPI avec Uvicorn.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
