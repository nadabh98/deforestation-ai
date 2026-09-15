
# 🌳 Deforestation AI V2 — Satellite Change Detection

> **End-to-end AI system for detecting potential deforestation from Sentinel-2 satellite imagery using deep learning, geospatial processing and Google Cloud.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-orange)](https://pytorch.org/)
[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run-blue)](https://cloud.google.com/run)
[![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-green)](https://earthengine.google.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED)](https://www.docker.com/)
[![Git](https://img.shields.io/badge/Version%20Control-Git-F05032)](https://git-scm.com/)

---

# 🚀 Live Application

## Interactive Dashboard

**Deforestation AI V2**

https://deforestation-dashboard-702314968420.europe-west1.run.app

The dashboard allows a user to:

- select an area directly on a satellite map,
- launch an AI analysis,
- automatically retrieve satellite imagery,
- visualize the predicted deforestation mask,
- visualize the probability map,
- obtain the percentage of potentially affected pixels.

**No satellite image upload is required.**

---

## REST API

**FastAPI backend**

https://deforestation-api-702314968420.europe-west1.run.app

The backend exposes the complete inference pipeline through a REST API deployed on Google Cloud Run.

---

# 🎯 Project Overview

**Deforestation AI V2** is an end-to-end **geospatial AI and computer vision application** designed to detect potential forest-cover changes from satellite imagery.

The project combines:

- 🛰️ Sentinel-2 multispectral satellite imagery
- 🌍 Google Earth Engine
- 🧠 PyTorch / U-Net
- 🖼️ Semantic segmentation
- 🐍 Python
- ⚡ FastAPI
- 📊 Streamlit
- 🐳 Docker
- ☁️ Google Cloud Run
- 🔧 Git / GitHub

The objective is not simply to train a machine-learning model.

The project demonstrates the complete AI engineering workflow:

```text
Satellite data
      ↓
Data acquisition
      ↓
Geospatial preprocessing
      ↓
ML dataset
      ↓
Model training
      ↓
Model evaluation
      ↓
Automated inference
      ↓
REST API
      ↓
Docker container
      ↓
Google Cloud Run
      ↓
Interactive dashboard
````

This project is aligned with **UN Sustainable Development Goal 15 — Life on Land**.

---

# 🌍 Problem Statement

Deforestation monitoring requires the analysis of large geographical areas and repeated satellite observations.

Traditional visual inspection of satellite imagery can be:

* time-consuming,
* difficult to scale,
* expensive for large areas,
* dependent on human interpretation.

Satellite imagery provides an opportunity to automate part of this process.

The objective of this project is therefore to investigate whether a deep-learning segmentation model can identify potential forest-cover changes between two satellite observations of the same geographical area.

The system compares:

```text
Historical reference image
             +
Recent satellite image
             ↓
       AI change detection
```

The result is a **pixel-level probability map** and a binary mask representing areas identified by the model as potential deforestation.

---

# 🧠 Machine Learning Approach

The core model is a **U-Net semantic segmentation network implemented with PyTorch**.

Unlike a simple image classifier, the model predicts a value for every pixel.

For each pixel:

```text
0.0 → low probability
1.0 → high probability
```

A threshold of:

```text
0.70
```

is then applied.

Pixels with:

```text
probability >= 0.70
```

are classified as potential deforestation.

---

# 🏗️ Model Architecture

```text
Reference Sentinel-2 image
        │
        ├── B2
        ├── B3
        ├── B4
        └── B8
              +
Recent Sentinel-2 image
        │
        ├── B2
        ├── B3
        ├── B4
        └── B8
              │
              ▼
       8-channel input
              │
              ▼
            U-Net
              │
              ▼
       Probability map
              │
              ▼
        Threshold 0.70
              │
              ▼
      Binary segmentation
              │
              ▼
    Deforestation statistics
```

---

# 📊 Model Configuration

| Parameter          | Value        |
| ------------------ | ------------ |
| Architecture       | U-Net        |
| Framework          | PyTorch      |
| Parameters         | 7,767,425    |
| Input channels     | 8            |
| Input resolution   | 256 × 256    |
| Output resolution  | 256 × 256    |
| Decision threshold | 0.70         |
| Inference device   | CPU          |
| Spatial resolution | 10 m / pixel |

---

# 🛰️ Satellite Data

The project uses **Sentinel-2 imagery** retrieved through **Google Earth Engine**.

Four spectral bands are used:

| Band | Description   | Resolution |
| ---- | ------------- | ---------- |
| B2   | Blue          | 10 m       |
| B3   | Green         | 10 m       |
| B4   | Red           | 10 m       |
| B8   | Near Infrared | 10 m       |

Each observation therefore contains:

```text
4 spectral bands
```

The reference and recent images are concatenated:

```text
4 bands reference
+
4 bands recent
=
8-channel model input
```

---

# 📐 Spatial Configuration

The V2 system analyzes a fixed maximum area of:

```text
2560 × 2560 meters
```

At a spatial resolution of:

```text
10 meters / pixel
```

this corresponds to:

```text
256 × 256 pixels
```

Therefore:

```text
256 pixels
×
10 meters
=
2560 meters
```

This ensures that the geographical analysis corresponds directly to the spatial dimensions expected by the model.

---

# 🔄 V1 → V2 Evolution

The project was developed incrementally.

## V1 — Initial AI Pipeline

The first version focused on validating the machine-learning approach.

The V1 workflow included:

```text
Satellite imagery
       ↓
Tile generation
       ↓
Training dataset
       ↓
U-Net
       ↓
Segmentation
       ↓
Evaluation
```

The initial dataset contained:

```text
528 initial tiles
```

After pairing and validation:

```text
280 image pairs
```

were used for:

```text
Training:   196
Validation: 42
Test:       42
```

The first model achieved:

```text
Dice:       0.8053
IoU:        0.6741
Precision:  0.7905
Recall:     0.8207
```

This validated the feasibility of the segmentation approach.

---

# 🚀 V2 — Automated Geospatial Inference

V2 focuses on transforming the research prototype into a usable AI application.

The major architectural improvement is:

> **The user no longer needs to provide satellite images.**

Instead, the user provides a geographical location.

The system automatically retrieves the required imagery from Google Earth Engine.

```text
User
 ↓
Select geographical area
 ↓
Latitude / Longitude
 ↓
Google Earth Engine
 ↓
Reference image
+
Recent image
 ↓
Preprocessing
 ↓
U-Net
 ↓
Prediction
 ↓
API
 ↓
Dashboard
```

---

# 🔄 V2 Automated Pipeline

```text
                    USER
                     │
                     ▼
             Select area on map
                     │
                     ▼
              Latitude / Longitude
                     │
                     ▼
           ┌─────────────────────┐
           │ Google Earth Engine │
           └──────────┬──────────┘
                      │
              ┌───────┴───────┐
              ▼               ▼
        Reference           Recent
        Sentinel-2          Sentinel-2
              │               │
              └───────┬───────┘
                      ▼
              GeoTIFF processing
                      │
                      ▼
              Spatial validation
                      │
                      ▼
             Band extraction
                      │
                      ▼
              Normalization
                      │
                      ▼
             8-channel tensor
                      │
                      ▼
                  U-Net
                      │
                      ▼
             Probability map
                      │
                      ▼
             Threshold = 0.70
                      │
                      ▼
              Binary mask
                      │
                      ▼
          Deforestation statistics
                      │
                      ▼
                  FastAPI
                      │
                      ▼
              Google Cloud Run
                      │
                      ▼
                Streamlit UI
```

---

# 📅 Temporal Configuration

The V2 system uses two temporal observations.

## Reference image

A historical Sentinel-2 observation is used as the reference state.

## Recent image

A recent Sentinel-2 observation is automatically searched within the configured recent period.

Current configuration:

```text
Recent observation window: 90 days
Maximum cloud coverage:    20%
```

The system therefore adapts the recent acquisition date according to the available satellite imagery.

---

# ☁️ Automatic Satellite Acquisition

One of the key improvements of V2 is the automatic satellite acquisition process.

The user does not need to:

* search for satellite images,
* download GeoTIFF files,
* crop the images,
* manually align the images,
* prepare the model input.

Instead:

```text
Coordinates
    ↓
Earth Engine search
    ↓
Suitable Sentinel-2 image
    ↓
Automatic export
    ↓
GeoTIFF
```

The system validates that the reference and recent images have compatible spatial extents before inference.

Small floating-point differences in raster bounds are handled using numerical tolerance rather than requiring exact binary equality.

---

# 🧪 Preprocessing Pipeline

The preprocessing stage performs several operations.

## 1. Raster loading

The Sentinel-2 GeoTIFF files are loaded using raster processing tools.

## 2. Band selection

Only the required bands are extracted:

```text
B2
B3
B4
B8
```

## 3. Spatial validation

The images are checked for:

* dimensions,
* spatial resolution,
* coordinate reference system,
* compatible spatial extent.

## 4. Normalization

The spectral values are normalized before being passed to the neural network.

## 5. Concatenation

The reference and recent images are concatenated along the channel dimension.

Final tensor:

```text
(1, 8, 256, 256)
```

---

# 🧠 Training

The model was trained using PyTorch.

The training configuration included:

```text
Optimizer:          Adam
Learning rate:      1e-4
Batch size:         2
Epochs:             15
Decision threshold: 0.70
```

The model uses a combined segmentation loss during training.

The best model is saved as:

```text
models/best_model.pth
```

The model contains:

```text
7,767,425 parameters
```

---

# 📊 Final Model Performance

The final evaluated model achieved:

| Metric    |      Score |
| --------- | ---------: |
| Dice      | **0.8582** |
| IoU       | **0.7517** |
| Precision | **0.8863** |
| Recall    | **0.8319** |

### Interpretation

**Dice = 0.8582**

Indicates strong overlap between predicted and reference segmentation masks.

**IoU = 0.7517**

Measures the intersection over union between predicted and reference regions.

**Precision = 0.8863**

Indicates that a large proportion of the pixels classified as deforestation correspond to the target class in the evaluation dataset.

**Recall = 0.8319**

Indicates that the model identifies a substantial proportion of the target pixels.

These values are dataset-specific and should not be interpreted as global deforestation-detection performance.

---

# 🧪 V2 Inference Example

Example geographical analysis:

```text
Latitude:   -3.119
Longitude:  -60.0217
Size:       2560 × 2560 m
```

The system automatically retrieved:

## Historical reference

```text
Date: 2020-07-29
Cloud coverage: ~0%
```

## Recent observation

```text
Date: 2026-08-26
Cloud coverage: ~0.05%
```

The model generated:

```text
Total pixels:       65,536
Deforested pixels:   2,551
Percentage:          3.89%
Threshold:           0.70
```

Therefore:

```text
Potential deforestation detected: 3.89%
```

This value represents the percentage of pixels classified by the model as potential deforestation within the analyzed area.

---

# 🗺️ Interactive Dashboard

The Streamlit dashboard is designed as the user-facing layer of the application.

## Geographic selection

The user can draw a rectangle directly on the map.

The dashboard retrieves:

```text
Latitude
Longitude
Width
Height
```

and converts the selected area into the parameters expected by the V2 API.

---

## Satellite map

The dashboard provides:

* satellite imagery,
* OpenStreetMap,
* interactive map controls,
* rectangle-based geographical selection.

---

## AI analysis

After selecting an area, the user launches the analysis.

The dashboard sends:

```json
{
  "latitude": -3.119,
  "longitude": -60.0217,
  "size": 2560,
  "recent_days": 90
}
```

to the FastAPI backend.

---

# 📊 Dashboard Results

The interface displays:

### Detection statistics

```text
Total pixels
Detected pixels
Deforestation percentage
```

### Model output

```text
Binary deforestation mask
```

### Probability output

```text
Pixel-level probability map
```

### Metadata

```text
Reference date
Recent date
Cloud coverage
Threshold
Analysis size
```

This makes the model output more interpretable than returning a raw prediction tensor alone.

---

# ⚡ FastAPI Backend

The backend exposes the AI pipeline through a REST API.

Main endpoint:

```text
POST /analyze-v2
```

Example request:

```json
{
  "latitude": -3.119,
  "longitude": -60.0217,
  "size": 2560,
  "recent_days": 90
}
```

The API:

1. validates the request,
2. retrieves satellite imagery,
3. launches the V2 pipeline,
4. runs model inference,
5. calculates statistics,
6. generates result files,
7. exposes the result through HTTP endpoints.

---

# 🧾 Example API Response

```json
{
  "job_id": "e4e68c7190f74ddd8f78bf641d9ae984",
  "latitude": -3.119,
  "longitude": -60.0217,
  "size_meters": 2560,
  "reference_date": "2020-07-29",
  "reference_cloud_percentage": 0.000498,
  "recent_date": "2026-08-26",
  "recent_cloud_percentage": 0.053788,
  "threshold": 0.7,
  "total_pixels": 65536,
  "deforested_pixels": 2551,
  "deforestation_percentage": 3.89251708984375
}
```

The API also exposes URLs for:

```text
Binary mask
Probability map
```

---

# ☁️ Google Cloud Architecture

The application is deployed using Google Cloud Run.

```text
                 Google Earth Engine
                         │
                         ▼
                  Sentinel-2 data
                         │
                         ▼
                AI Processing Pipeline
                         │
                         ▼
                    U-Net Model
                         │
                         ▼
                  FastAPI Backend
                         │
                         ▼
                  Google Cloud Run
                         │
                         ▼
                Streamlit Dashboard
                         │
                         ▼
                       User
```

The system is composed of two independently deployed services:

```text
deforestation-api
        +
deforestation-dashboard
```

---

# 🐳 Containerization

Both services are containerized.

## Backend

The root `Dockerfile` packages the API and AI inference environment.

## Dashboard

The dashboard has its own Docker configuration:

```text
dashboard/
├── Dockerfile
├── app.py
└── requirements.txt
```

This allows the frontend and backend to be deployed independently.

---

# 🚀 Deployment

## Backend

```bash
gcloud run deploy deforestation-api \
  --source . \
  --region europe-west1 \
  --project deforestation-ai-projet \
  --allow-unauthenticated
```

## Dashboard

```bash
gcloud run deploy deforestation-dashboard \
  --source dashboard \
  --region europe-west1 \
  --project deforestation-ai-projet \
  --allow-unauthenticated \
  --set-env-vars API_URL=https://deforestation-api-702314968420.europe-west1.run.app
```

---

# 🧪 Local Development

## Clone

```bash
git clone git@github.com:nadabh98/deforestation-ai.git
cd deforestation-ai
```

## Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the API

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000
```

---

# ▶️ Run the Dashboard

```bash
streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

---

# 🔬 Run V2 Pipeline Locally

Example:

```bash
python v2/scripts/pipeline_v2.py \
  --lat -3.119 \
  --lon -60.0217 \
  --size 2560 \
  --days 90
```

Expected workflow:

```text
1. Retrieve reference image
2. Retrieve recent image
3. Validate imagery
4. Prepare model input
5. Run U-Net
6. Generate probability map
7. Generate binary mask
8. Calculate statistics
9. Save results
```

---

# 📂 Complete Project Structure

```text
deforestation-ai/
│
├── api/
│   └── main.py
│
├── dashboard/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── models/
│   └── best_model.pth
│
├── v2/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── sentinel2_reference.tif
│   │   │   └── sentinel2_recent.tif
│   │   │
│   │   ├── processed/
│   │   │   └── inference_input.pt
│   │   │
│   │   └── ml_dataset/
│   │
│   ├── reports/
│   │
│   └── scripts/
│       ├── download_sentinel2.py
│       ├── earth_engine_v2.py
│       ├── pipeline_v2.py
│       ├── prepare_v2_inference.py
│       └── run_v2_inference.py
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧩 V2 Script Responsibilities

## `download_sentinel2.py`

Responsible for automatically retrieving Sentinel-2 imagery.

It handles:

* Earth Engine image search,
* cloud filtering,
* reference image selection,
* recent image selection,
* export of satellite imagery.

---

## `earth_engine_v2.py`

Contains the Earth Engine integration used by the V2 pipeline.

It manages the interaction between:

```text
Python
   ↓
Google Earth Engine
   ↓
Sentinel-2
```

---

## `pipeline_v2.py`

Main orchestration layer.

It connects:

```text
Download
   ↓
Preprocessing
   ↓
Inference
   ↓
Statistics
   ↓
Results
```

This is the central V2 pipeline.

---

## `prepare_v2_inference.py`

Transforms the downloaded satellite images into the tensor expected by the U-Net.

Output:

```text
v2/data/processed/inference_input.pt
```

---

## `run_v2_inference.py`

Loads the trained model and performs inference.

It generates:

```text
Probability map
Binary mask
```

and calculates:

```text
Detected pixels
Total pixels
Deforestation percentage
```

---

# 🗃️ Git and Repository Management

The project uses Git for version control.

The V2 implementation is maintained on a dedicated branch:

```text
v2-single-image
```

The repository excludes large generated files and model artifacts.

Examples:

```text
*.tif
*.tiff
v2/data/
v2/reports/results/
models/*.pth
```

This prevents large binary datasets and generated outputs from unnecessarily increasing repository size.

---

# 🔐 Configuration and Security

Sensitive configuration files are excluded from version control.

The repository ignores:

```text
.env
*.key
```

Credentials and cloud authentication are therefore handled outside the source repository.

---

# 📦 Reproducibility

The project separates:

```text
Source code
      ↓
Model
      ↓
Configuration
      ↓
Generated data
```

Generated satellite imagery and inference results are not committed to Git.

This keeps the repository lightweight while preserving the complete processing logic required to reproduce the pipeline in an appropriately configured environment.

---

# ⚠️ Scientific Limitations

This project is a **research and portfolio project**.

It should not be interpreted as an operational or authoritative deforestation-monitoring system.

## Satellite limitations

Sentinel-2 imagery can be affected by:

* clouds,
* haze,
* atmospheric conditions,
* seasonal changes,
* acquisition conditions.

## Model limitations

The model can potentially classify other land-cover changes as deforestation.

Possible sources of false positives include:

* agriculture,
* vegetation management,
* fires,
* temporary vegetation loss,
* seasonal variations,
* other land-use changes.

## Geographic generalization

The model has not been validated across all ecosystems and geographical regions.

The reported metrics therefore describe the evaluation dataset used during development.

They should not be interpreted as global performance indicators.

## Interpretation

The result should therefore be interpreted as:

> **Potential forest-cover change detected by the trained AI model.**

It does not constitute independent confirmation of actual or illegal deforestation.

---

# 🔮 Future Improvements

## Data

* [ ] Expand the training dataset
* [ ] Add more geographical regions
* [ ] Increase temporal diversity
* [ ] Improve annotation quality
* [ ] Add additional Sentinel-2 bands

## AI

* [ ] Compare U-Net with U-Net++
* [ ] Evaluate Transformer-based segmentation
* [ ] Improve temporal change detection
* [ ] Add model explainability
* [ ] Calibrate prediction confidence
* [ ] Perform cross-region validation

## Geospatial

* [ ] Integrate Sentinel-1 SAR
* [ ] Add multi-temporal analysis
* [ ] Build historical change maps
* [ ] Support larger areas
* [ ] Automate region monitoring

## Cloud / MLOps

* [ ] Add asynchronous job processing
* [ ] Add model versioning
* [ ] Add monitoring
* [ ] Add automated evaluation
* [ ] Add CI/CD
* [ ] Add scalable batch inference
* [ ] Add production observability

---

# 🌱 Sustainable Development Goal 15

This project contributes to:

## UN Sustainable Development Goal 15 — Life on Land

The objective is to explore how:

```text
Earth Observation
        +
Artificial Intelligence
        +
Cloud Computing
```

can support the monitoring of terrestrial ecosystems.

---

# 🛠️ Technology Stack

| Category            | Technology            |
| ------------------- | --------------------- |
| Language            | Python 3.12           |
| Deep Learning       | PyTorch               |
| Architecture        | U-Net                 |
| Computer Vision     | Semantic Segmentation |
| Satellite           | Sentinel-2            |
| Geospatial Platform | Google Earth Engine   |
| Raster Processing   | Rasterio              |
| Numerical Computing | NumPy                 |
| API                 | FastAPI               |
| Frontend            | Streamlit             |
| Containerization    | Docker                |
| Cloud Platform      | Google Cloud          |
| Serverless Compute  | Cloud Run             |
| Version Control     | Git / GitHub          |

---

# 💡 Engineering Highlights

This project demonstrates experience across several layers of AI engineering.

## Machine Learning

* Semantic segmentation
* U-Net architecture
* Model training
* Model evaluation
* Probability thresholding
* Inference pipeline

## Computer Vision

* Pixel-level segmentation
* Multispectral imagery
* Image comparison
* Spatial change detection

## Geospatial AI

* Sentinel-2
* GeoTIFF
* Coordinate systems
* Spatial resolution
* Google Earth Engine
* Automated satellite data acquisition

## Backend Engineering

* REST API
* FastAPI
* Request validation
* Inference orchestration
* Result serving

## Cloud Engineering

* Docker
* Google Cloud Run
* Independent frontend/backend deployment
* Cloud-hosted inference

## Application Engineering

* Interactive geospatial dashboard
* API integration
* Result visualization
* User-oriented AI workflow

---

# 📈 From Research Prototype to AI Application

The main evolution of this project can be summarized as:

```text
                     V1
                      │
                      ▼
            Validate AI approach
                      │
                      ▼
              Train U-Net model
                      │
                      ▼
               Evaluate model
                      │
                      ▼
                     V2
                      │
                      ▼
          Automate satellite retrieval
                      │
                      ▼
          Automate preprocessing
                      │
                      ▼
           Automate model inference
                      │
                      ▼
                Build REST API
                      │
                      ▼
               Dockerize system
                      │
                      ▼
              Deploy to Cloud Run
                      │
                      ▼
             Build interactive UI
                      │
                      ▼
             Complete AI product
```

The project therefore moves beyond a standalone machine-learning experiment toward a complete **deployed AI system**.

---

# 🌐 End-to-End Architecture

```text
┌──────────────────────────────────────────────┐
│                  USER                        │
│                                              │
│        Select geographical area              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              STREAMLIT UI                   │
│                                              │
│       Interactive satellite map             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│               FASTAPI API                   │
│                                              │
│        Request validation & orchestration    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│          GOOGLE EARTH ENGINE                │
│                                              │
│       Sentinel-2 image acquisition          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│            PREPROCESSING                    │
│                                              │
│      4 bands + 4 bands = 8 channels         │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                 U-NET                      │
│                                              │
│        Pixel-level segmentation             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│               RESULTS                       │
│                                              │
│      Mask + Probability + Statistics        │
└──────────────────────────────────────────────┘
```

---

# 🧭 End-to-End Workflow

A complete user request follows this sequence:

```text
1. User opens the Streamlit dashboard
                    ↓
2. User selects a geographical area
                    ↓
3. Dashboard validates the selected zone
                    ↓
4. Dashboard sends coordinates to FastAPI
                    ↓
5. FastAPI launches the V2 pipeline
                    ↓
6. Google Earth Engine searches Sentinel-2
                    ↓
7. Reference and recent images are retrieved
                    ↓
8. Images are exported as GeoTIFF
                    ↓
9. Spatial and spectral preprocessing
                    ↓
10. Reference + recent images are concatenated
                    ↓
11. U-Net performs semantic segmentation
                    ↓
12. Probability map is generated
                    ↓
13. Threshold 0.70 produces binary mask
                    ↓
14. Deforestation statistics are calculated
                    ↓
15. Results are exposed through the API
                    ↓
16. Dashboard displays the results
```

---

# 🎓 What This Project Demonstrates

This project demonstrates the ability to work across the complete AI development lifecycle:

```text
Data Engineering
       +
Machine Learning
       +
Computer Vision
       +
Geospatial Processing
       +
Backend Development
       +
Cloud Engineering
       +
Application Development
```

It demonstrates not only the ability to train an AI model, but also to **integrate that model into a usable, cloud-deployed application**.

---

# 🌟 Key Takeaways

### From data to model

Sentinel-2 multispectral imagery is transformed into an 8-channel deep-learning input.

### From model to inference

The U-Net produces pixel-level probability predictions and segmentation masks.

### From inference to API

The complete inference pipeline is exposed through FastAPI.

### From API to cloud

The backend is containerized and deployed to Google Cloud Run.

### From cloud to user

A Streamlit dashboard provides an interactive geospatial interface.

### Final result

```text
Satellite Data
      ↓
AI Model
      ↓
Inference Pipeline
      ↓
REST API
      ↓
Cloud Deployment
      ↓
Interactive Application
```

---

# 👩‍💻 Author

## Nada BEN HAMMOUDA

**Biomedical Engineer | AI Engineer**

Areas of interest:

* Biomedical AI
* Computer Vision
* Machine Learning
* Geospatial AI
* Cloud AI
* Generative AI
* AI applications in healthcare
* AI applications in environmental monitoring

---

# 📫 Project Links

## GitHub Repository

[https://github.com/nadabh98/deforestation-ai](https://github.com/nadabh98/deforestation-ai)

## Live Dashboard

[https://deforestation-dashboard-702314968420.europe-west1.run.app](https://deforestation-dashboard-702314968420.europe-west1.run.app)

## FastAPI Backend

[https://deforestation-api-702314968420.europe-west1.run.app](https://deforestation-api-702314968420.europe-west1.run.app)

---

# ⭐ Project Summary

**Deforestation AI V2** is an end-to-end geospatial AI application that transforms satellite observations into an automated forest-cover change detection workflow.

```text
             SATELLITE DATA
                    │
                    ▼
          GOOGLE EARTH ENGINE
                    │
                    ▼
           GEOSPATIAL PROCESSING
                    │
                    ▼
             DEEP LEARNING
                    │
                    ▼
                U-NET
                    │
                    ▼
          SEMANTIC SEGMENTATION
                    │
                    ▼
              FASTAPI API
                    │
                    ▼
                DOCKER
                    │
                    ▼
          GOOGLE CLOUD RUN
                    │
                    ▼
             STREAMLIT UI
                    │
                    ▼
              END USER
```

> **From raw satellite imagery to a deployed AI application for automated forest-cover change detection.**

```
```

