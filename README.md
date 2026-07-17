# 🛠️ Real-Time Industrial Defect Detection System

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.110.0%2B-009688.svg?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-FF6F00.svg?style=flat-square&logo=target)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat-square&logo=docker)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg?style=flat-square&logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Metrics-F46800.svg?style=flat-square&logo=grafana)](https://grafana.com/)

An end-to-end, edge-deployable computer vision system for real-time detection and localization of manufacturing surface defects on metal sheets. Built using **FastAPI**, **YOLOv8**, and **OpenCV**, with containerized deployment and performance monitoring.

---

## 📌 Project Objective & Highlights

In manufacturing, detecting surface anomalies is critical for quality control. This system provides:
* ⚡ **Real-Time Edge Inference**: Low latency, frame-by-frame object detection for image uploads, video streams, and live camera feeds.
* 🤖 **High Accuracy Defect Classification**: Identifies six distinct surface defect classes using an optimized YOLOv8 model.
* 📈 **Class-Aware Augmentation**: Balances minority classes offline using a specialized Albumentations pipeline.
* 📊 **Production Observability**: Built-in API latency, system resource, and prediction metrics exported to Prometheus and visualized on a Grafana dashboard.

---

## 🛠️ Tech Stack & Layers

```
                                  ┌───────────────────────────┐
                                  │      Client / Frontend    │
                                  └─────────────┬─────────────┘
                                                │ REST / Streams
                                                ▼
     ┌─────────────────────────────────────────────────────────────────────────────────────┐
     │                                   Backend API Layer                                 │
     │  ┌───────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────┐  │
     │  │    FastAPI Server     │──>│ Preprocessing (OpenCV / RGB)│──>│  YOLOv8 Service │  │
     │  │ (Uvicorn / Pydantic)  │   │   - Validation & Resizing   │   │  (Ultralytics)  │  │
     │  └──────────┬────────────┘   └─────────────────────────────┘   └────────┬────────┘  │
     └─────────────┼───────────────────────────────────────────────────────────┼───────────┘
                   │ Metrics Pull                                              │ Predictions
                   ▼                                                           ▼
     ┌──────────────────────────┐                                   ┌───────────────────┐
     │    Prometheus Server     │                                   │  Trained Weights  │
     └─────────────┬────────────┘                                   │   (best/last.pt)  │
                   │ Visualize                                      └───────────────────┘
                   ▼
     ┌──────────────────────────┐
     │    Grafana Dashboards    │
     └──────────────────────────┘
```

* **Core Language**: Python 3.12+
* **Data Processing & Augmentation**: OpenCV, Albumentations, NumPy, Matplotlib
* **Deep Learning Framework**: PyTorch, Ultralytics YOLOv8
* **REST API Layer**: FastAPI, Uvicorn, Pydantic (v2), python-multipart
* **Deployment & Monitoring**: Docker, Docker Compose, Prometheus, Grafana

---

## 📁 Repository Structure

```
├── backend/                  # FastAPI REST API implementation
│   ├── app/                  # Application source code
│   │   ├── api/              # API router and endpoints (/predict, /health, /root)
│   │   ├── core/             # Configuration, logging, and settings management
│   │   ├── models/           # YOLO model wrapper and service loaders
│   │   ├── schemas/          # Pydantic request and response validation schemas
│   │   └── services/         # Image preprocessing and model inference services
│   ├── logs/                 # Rotating backend application logs
│   ├── tests/                # Endpoint validation and API integration tests
│   └── requirements.txt      # Backend-specific package dependencies
├── configs/                  # Global YAML configuration files
│   ├── data.yaml             # YOLOv8 dataset splits and class maps
│   ├── classes.yaml          # Defect names, IDs, and descriptions
│   ├── experiment.yaml       # Training training hyperparameters
│   ├── augmentation.yaml     # Albumentations offline transform parameters
│   └── prometheus.yml        # Metrics scraping configuration
├── dataset/                  # Active training/validation dataset (YOLO format)
├── datasets/                 # Raw NEU-DET dataset storage (Pascal VOC format)
├── docs/                     # Technical specifications and guides
├── exports/                  # Compiled model exports (ONNX / TensorRT)
├── inference/                # Dedicated offline/live camera test scripts
├── models/                   # Local PyTorch weights cache
├── reports/                  # Pipeline quality, validation, and analytics reports
│   └── graphs/               # Auto-generated charts and distribution plots
├── results/                  # Local training metrics, learning curves, confusion matrices
├── scripts/                  # Automated data preparation, balancing, and report utility scripts
├── tests/                    # Core unit and integration test suites
├── training/                 # Model training, evaluation, and prediction loops
├── utils/                    # Shared helper modules (data loader, visualization, device info)
├── weights/                  # Pretrained and custom YOLO weights (.pt)
├── docker-compose.yml        # Orchestration for API, Prometheus, and Grafana
└── Dockerfile                # API container definition (Python 3.12 slim)
```

---

## 📊 Dataset & Class Taxonomy

The system is trained on the **NEU Metal Surface Defects Database (NEU-DET)** which contains hot-rolled steel strip surface defects:

| Class ID | Defect Name | Code | Description |
| :---: | :--- | :---: | :--- |
| **0** | Crazing | `cr` | Network-like micro-cracking on the metal surface |
| **1** | Inclusion | `in` | Non-metallic particles embedded in the surface |
| **2** | Patches | `pa` | Localized sheet metal surface imperfections |
| **3** | Pitted Surface | `ps` | Small pits or depressions caused by roll wear |
| **4** | Rolled-in Scale | `rs` | Iron oxides pressed into the metal surface during rolling |
| **5** | Scratches | `sc` | Linear abrasions caused by mechanical friction |

---

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and run the setup script to create a virtual environment and install dependencies:

```powershell
# Run the PowerShell setup script (Windows)
.\scripts\setup_environment.ps1

# Or install manually
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Data Pipeline

Prepare, inspect, balance, preprocess, and analyze the dataset using the script pipeline:

```bash
# Step 1: Verify raw dataset integrity
python scripts/verify_raw_dataset.py
 
# Step 2: Convert VOC XML to YOLO TXT format and split
python scripts/convert_to_yolo.py

# Step 3: Run automated dataset health monitoring (generates reports/dataset_health/dataset_health_report.md)
python scripts/monitor_dataset_health.py

# Step 4: Run automated dataset quality inspection (generates reports/quality/dataset_quality_report.md)
python scripts/inspect_dataset_quality.py
 
# Step 5: Run comprehensive pair, annotation, and advanced box/overlap validation (generates reports/dataset_validation_report.md and reports/dataset_integrity_report.md)
python scripts/verify_dataset.py
  
# Step 6: Balance class distribution using Albumentations offline augmentation (generates reports/dataset_balancing_report.md)
python scripts/augment_dataset.py

# Step 7: Preprocess split images concurrently (resizing, normalising, corruption check)
python scripts/preprocess_pipeline.py

# Step 8: Run automated dataset anomaly detection (generates reports/anomalies/anomaly_report.md)
python scripts/detect_anomalies.py

# Step 9: Benchmark preprocessing transforms (generates reports/preprocessing_benchmark.md)
python scripts/benchmark_preprocessing.py

# Step 10: Compute composite dataset quality score and generate distribution charts (generates reports/dataset_quality_report.md)
python scripts/dataset_quality_score.py

# Step 11: Generate dataset analytics dashboard and visualizations (generates reports/analytics/dataset_analytics_report.md)
python scripts/generate_dataset_dashboard.py

# Step 12: Track dataset versions and generate metadata database (generates dataset/metadata.json)
python scripts/manage_metadata.py
```

### 3. Model Training & Evaluation

The training pipeline provides full CLI configuration, automatic results parsing, learning curve plotting, and metric report generation.

```bash
# Run a training dry-run (1 epoch, minimal settings) to verify pipeline integrity
python training/train.py --dry-run

# Run full model training (GPU recommended)
python training/train.py --epochs 100 --batch 16 --device cuda

# Evaluate the model on the validation split
python training/evaluate.py --model results/training/weights/best.pt

# Run prediction inference on custom images
python training/predict.py --model results/training/weights/best.pt --source dataset/yolo/images/test/
```

Training metrics, loss curves, confusion matrices, and formatted markdown evaluation reports are automatically saved under the timestamped directory: `results/`.

---

## ⚡ FastAPI Backend API

The REST API exposes prediction endpoints for images, videos, and live feeds.

### Endpoints

| Method | Route | Description | Input / Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root Welcome Endpoint | JSON with application metadata |
| `GET` | `/health` | Application Health Status | JSON with CPU/Memory usage metrics |
| `POST` | `/predict/image` | Image Defect Detection | Multipart Form (`file`: Image upload) |
| `POST` | `/predict/video` | Video Frame-by-Frame Inference | Multipart Form (`file`: Video upload) |
| `GET` | `/predict/video/download/{filename}` | Download Annotated Video File | Serve annotated output video |
| `POST` | `/predict/live` | Live Camera Stream Validation | JSON Body (`source`: RTSP/RTMP URL or Device Index) |
| `GET` | `/metrics` | Prometheus Metrics Endpoint | Prom-format raw metrics data |
| `GET` | `/docs` | OpenAPI Swagger UI | Interactive documentation |

### Running the API Locally

```bash
cd backend
# Create environment configurations
cp .env.example .env

# Start the uvicorn development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Deployment & Monitoring Stack

The project features a containerized multi-container setup containing:
* **API Service**: FastAPI application running on port `8000`.
* **Prometheus**: Metrics collection server running on port `9090`.
* **Grafana**: Interactive visualization dashboard running on port `3000`.

### Deploying the Stack

```bash
# Start all containers in detached mode
docker-compose up -d

# Check running container statuses
docker-compose ps

# Tear down the stack and delete anonymous volumes
docker-compose down -v
```

### Observability Dashboard Configuration
1. Access the Grafana UI at `http://localhost:3000` (Default Credentials: `admin` / `admin`).
2. Add **Prometheus** as a data source with the URL `http://prometheus:9090`.
3. Create charts using metrics exported by the backend, such as `http_requests_total` and API request latency histograms.

### Production Security & Deployment Details
For detailed step-by-step instructions on securing the API endpoints using **API Key Authentication**, verifying configurations, and deploying to production, refer to the [Production Deployment & Configuration Guide](docs/deployment.md).

---

## 📋 Phase Tracker

| Phase | Milestone | Status |
| :---: | :--- | :---: |
| **Data** | Stratified split & VOC XML → YOLO conversion | ✅ Complete |
| **Data** | Image integrity, duplicate, and annotations validation | ✅ Complete |
| **Data** | Class-aware Albumentations offline augmentation & balancing | ✅ Complete |
| **Data** | Automated report suite & class distribution graphs | ✅ Complete |
| **ML** | Configurable training pipeline supporting overrides & dry-runs | ✅ Complete |
| **ML** | Automatic evaluation reports, metrics parsing, and curve plotting | ✅ Complete |
| **API** | Image, video, and live stream inference API endpoints | ✅ Complete |
| **Deployment**| Multi-container Docker Compose configuration | ✅ Complete |
| **Deployment**| Prometheus & Grafana metrics instrumentation | ✅ Complete |
| **Integration**| Real weights training & mock model replacement | ✅ Complete |
| **Integration**| ONNX / TensorRT runtime model compilation | ✅ Complete |
| **Integration**| Frontend Dashboard UI & Real Webcam integration | ✅ Complete |

---

## 👥 Collab Contribution Roles

* **Druva** (ML Engineer): Authored the training, evaluation, and prediction pipeline logic, data loader utilities, device helpers, configuration manager, metric parsers, and plotting libraries.
* **Saniya** (CV & Data Engineer): Built the dataset preprocessing, verification, duplicate detection, pair validation, offline augmentation, and automated markdown reporting suite.
* **Prajwal** (Backend & Deployment Engineer): Implemented the FastAPI service architecture, image/video/live endpoints, logging/exception handlers, Dockerfile/Docker Compose configs, and Prometheus instrumentation.
