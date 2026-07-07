# Training Guide — Real-Time Industrial Defect Detection System

> **Author:** saniyamirjanavar-hash  
> **Date:** 2026-07-07  
> **Model:** YOLOv8 (Ultralytics)  
> **Dataset:** NEU Metal Surface Defects (6 classes)

---

## 1. Prerequisites

### Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `ultralytics` | ≥ 8.0 | YOLOv8 training & inference |
| `opencv-python` | ≥ 4.8 | Image processing |
| `albumentations` | ≥ 1.3 | Data augmentation |
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Visualization |
| `PyYAML` | ≥ 6.0 | Config file parsing |

---

## 2. Dataset Preparation

Run the full dataset finalization pipeline before training:

```bash
# Step 1 — Full dataset build (validates, fixes IDs, writes data.yaml)
python scripts/build_final_dataset.py

# Step 2 — Re-run augmentation if needed (tops up minority classes)
python scripts/augment_dataset.py

# Step 3 — Generate class distribution charts
python scripts/dataset_stats.py

# Step 4 — Final pair validation (should exit 0)
python scripts/validate_pairs.py
```

### Expected Dataset Counts After Build

| Split | Images | Notes |
|-------|--------|-------|
| Train | 1,931 | includes augmented `aug_*` images |
| Val   |   361 | original images only |
| Test  |   181 | original images only |

---

## 3. Model Training

### Quick Start

```bash
# Train YOLOv8n (nano) — fastest, good for testing
yolo detect train \
    data=configs/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=neu_defect_nano \
    project=results

# Train YOLOv8s (small) — recommended baseline
yolo detect train \
    data=configs/data.yaml \
    model=yolov8s.pt \
    epochs=150 \
    imgsz=640 \
    batch=16 \
    name=neu_defect_small \
    project=results
```

### Via Python Training Script

```bash
# Standard training
python training/train.py --epochs 150 --batch_size 16 --device cuda

# CPU-only training
python training/train.py --epochs 50 --batch_size 8 --device cpu

# Quick pipeline sanity check (no actual training)
python training/train.py --dry-run
```

### Recommended Hyperparameters

```yaml
# configs/hyperparameters.yaml (key values)
lr0:          0.01      # Initial learning rate
lrf:          0.01      # Final LR (cosine annealing factor)
momentum:     0.937
weight_decay: 0.0005
warmup_epochs: 3.0
box:          7.5       # Box loss gain
cls:          0.5       # Class loss gain
dfl:          1.5       # Distribution focal loss gain
```

---

## 4. Evaluation

```bash
# Evaluate on validation split
python training/evaluate.py --split val --device cpu

# Evaluate on test split
python training/evaluate.py --split test --device cpu

# Or via yolo CLI
yolo detect val \
    data=configs/data.yaml \
    model=results/neu_defect_small/weights/best.pt \
    split=test
```

### Expected Metrics (YOLOv8s, 150 epochs)

| Metric | Expected Range |
|--------|---------------|
| mAP@50 | 0.70 – 0.85 |
| Precision | 0.75 – 0.90 |
| Recall | 0.70 – 0.85 |

---

## 5. Inference

```bash
# Single image
python training/predict.py \
    --source dataset/yolo/images/test/crazing_1.jpg \
    --conf 0.4 \
    --weights results/neu_defect_small/weights/best.pt

# Directory of images
python training/predict.py \
    --source dataset/yolo/images/test/ \
    --conf 0.4

# Via yolo CLI
yolo detect predict \
    model=results/neu_defect_small/weights/best.pt \
    source=dataset/yolo/images/test/ \
    conf=0.4 \
    save=True
```

---

## 6. Unit Tests

```bash
# Run all dataset validation tests
python -m unittest tests/test_dataset_validation.py -v

# Run training pipeline tests (dry-run mode)
python -m unittest tests/test_train_pipeline.py -v

# Run prediction / evaluation tests
python -m unittest tests/test_predict.py tests/test_evaluate.py -v
```

---

## 7. Project Directory Reference

```
Real-Time-Industrial-Defect-Detection-System/
├── configs/
│   ├── data.yaml              # YOLOv8 dataset config  ← auto-updated by build script
│   ├── augmentation.yaml      # Albumentations pipeline config
│   ├── hyperparameters.yaml   # Training hyperparameters
│   └── experiment.yaml        # Experiment tracking
├── dataset/
│   └── yolo/
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── scripts/
│   ├── config.py              # Shared constants & paths (import this)
│   ├── build_final_dataset.py # Master pipeline orchestrator
│   ├── augment_dataset.py     # Albumentations augmentation + balancing
│   ├── validate_pairs.py      # Image-label pair validator
│   ├── verify_dataset.py      # Folder structure & corruption check
│   ├── dataset_stats.py       # Class distribution analysis + charts
│   ├── generate_reports.py    # Markdown quality reports
│   └── optimize_preprocess.py # Duplicate detection + annotation QA
├── training/
│   ├── train.py               # YOLOv8 training entry point
│   ├── predict.py             # Inference pipeline
│   └── evaluate.py            # Metrics evaluation
├── reports/
│   ├── graphs/                # PNG charts
│   ├── class_distribution.md
│   ├── dataset_quality_report.md
│   ├── preprocessing_validation_summary.md
│   ├── validation_report.md
│   └── final_dataset_summary.md
├── docs/
│   ├── dataset.md             # Dataset documentation (this file's companion)
│   └── TRAINING_GUIDE.md      # This file
└── tests/
```

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `data.yaml` path error | Run `python scripts/build_final_dataset.py` to regenerate |
| `CUDA out of memory` | Reduce `--batch_size` or use `--device cpu` |
| Low mAP on `inclusion` | Class may still be underrepresented — re-run `augment_dataset.py` |
| `cv2.imread` returns None | Image is corrupted — run `verify_dataset.py` to identify |
| Float class IDs in labels | Run `python scripts/optimize_preprocess.py` to auto-fix |
| Missing label files | Check `reports/validation_report.md` for a full list |
