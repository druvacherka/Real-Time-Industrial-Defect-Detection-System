# ML Development & Training Framework Guide

This document describes how to use the Day 3 training pipeline skeletons, annotation visualization utility, and tests.

---

## 🎨 Annotation Visualization Utility

To check raw image annotations visually, use [visualize_annotations.py](file:///c:/Users/druva/projects/Real-Time-Industrial-Defect-Detection-System/scripts/visualize_annotations.py). It parses Pascal VOC XML format annotation files, draws class-colored bounding boxes on images, and saves results.

### Usage
```bash
# Run inside the virtual environment
.\venv\Scripts\python.exe scripts/visualize_annotations.py --split train --num_samples 3
```

### Options
* `--split`: Set to `train` or `validation` (defaults to `train`).
* `--num_samples`: Number of random samples to visualize.
* `--output_dir`: Directory where visualized images are saved (defaults to `results/visualizations/`).

---

## 🏃 ML Training & Inference Skeletons

The core pipeline scripts reside in the `training/` folder:

### 1. Model Training Skeleton
[train.py](file:///c:/Users/druva/projects/Real-Time-Industrial-Defect-Detection-System/training/train.py) initializes YOLOv8 model configs and defaults from [configs/experiment.yaml](file:///c:/Users/druva/projects/Real-Time-Industrial-Defect-Detection-System/configs/experiment.yaml).
```bash
.\venv\Scripts\python.exe training/train.py --epochs 50 --batch_size 16 --device cpu
```

### 2. Inference / Prediction Skeleton
[predict.py](file:///c:/Users/druva/projects/Real-Time-Industrial-Defect-Detection-System/training/predict.py) runs object detection on target image inputs.
```bash
.\venv\Scripts\python.exe training/predict.py --source datasets/raw/NEU-DET/train/images/crazing/crazing_1.jpg --conf 0.4
```

### 3. Evaluation Skeleton
[evaluate.py](file:///c:/Users/druva/projects/Real-Time-Industrial-Defect-Detection-System/training/evaluate.py) computes metrics (mAP, Precision, Recall) on selected dataset splits.
```bash
.\venv\Scripts\python.exe training/evaluate.py --split val --device cpu
```

---

## 🧪 Running Unit Tests

Ensure all tests pass prior to pushing changes:
```bash
.\venv\Scripts\python.exe -m unittest tests/test_visualization.py
.\venv\Scripts\python.exe -m unittest tests/test_skeletons.py
```
