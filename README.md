# Real-Time Industrial Defect Detection System

A real-time computer vision system for detecting industrial surface defects using **YOLOv8**.

---

## Project Objective

Build an edge-deployable defect detection system capable of identifying manufacturing defects from live camera feeds with high accuracy and low latency.

---

## Dataset

- **NEU Metal Surface Defects Database**
- Six defect classes:
  - Crazing (cr)
  - Inclusion (in)
  - Patches (pa)
  - Pitted Surface (ps)
  - Rolled-in Scale (rs)
  - Scratches (sc)

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- ONNX Runtime
- NVIDIA TensorRT
- FastAPI

---

## Repository Structure

Below is the directory structure layout for this project and the explanation of each folder:

* **`dataset/`**: The complete dataset preparation pipeline directory.
  * `raw/`: Stores raw unaltered images and XML annotations.
  * `processed/`: Standardized format dataset storage.
  * `yolo/`: Organized splits (`train/`, `val/`, `test/`) with `images/` and `labels/` for YOLOv8 model training.
  * `annotations/`: Intermediate annotations (e.g. converted JSON or CSV files).
  * `augmented/`: Output folder for offline data augmentation.
  * `reports/`: Folder containing statistics, verification outputs, and distribution charts.
* **`configs/`**: Holds configuration YAML files (`data.yaml`, `classes.yaml`, `augmentation.yaml`, `experiment.yaml`) controlling dataset mappings, training parameters, and augmentation pipelines.
* **`docs/`**: Project documentation, including standard data workflows and setup details (`dataset.md`, `development.md`).
* **`scripts/`**: Executable Python scripts for automation (`verify_dataset.py`, `dataset_stats.py`, `visualize_samples.py`).
* **`exports/`**: Contains compiled model formats (e.g. ONNX, TensorRT engines) optimized for production environments.
* **`inference/`**: Houses modules for running local and API-based inference tasks.
* **`models/`**: Defines custom neural network configurations or model wrappers.
* **`notebooks/`**: Jupyter notebooks for exploratory data analysis (EDA), prototype training, and feature testing.
* **`results/`**: Training progress checkpoints, metrics logs, confusion matrices, and model validation plots.
* **`tests/`**: Unit and integration test suites for software validation.
* **`training/`**: Training scripts and loss optimization pipelines.
* **`utils/`**: Helper methods for logging, file systems, and image helpers.
* **`weights/`**: Directory storing model checkpoints (`.pt` files).

---

## Scripts Usage

### 1. Dataset Verification
Run the following script to check the integrity of YOLO splits (missing images/labels, empty directories):
```bash
python scripts/verify_dataset.py
```

### 2. Dataset Statistics
Run this script to calculate the distribution of images, labels, and class instances across the splits:
```bash
python scripts/dataset_stats.py
```

### 3. Sample Visualization
Run this script to display random dataset samples complete with bounding box overlays:
```bash
python scripts/visualize_samples.py
```

---

## Team

- ML Engineer
- Computer Vision & Data Engineer
- Backend & Deployment Engineer

---

## Current Status

🚧 Project Initialization and Dataset Pipeline Setup Completed