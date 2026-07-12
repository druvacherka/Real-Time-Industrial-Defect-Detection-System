# NEU Metal Surface Defects — Dataset Documentation

> **Last updated:** 2026-07-12  
> **Author:** saniyamirjanavar-hash (Senior Computer Vision & Data Engineer)

---

## Dataset Overview

The **NEU Metal Surface Defects Dataset** (NEU-DET) was published by Northeastern University and is widely used as a benchmark for automatic surface defect detection on hot-rolled steel strips.

| Property | Value |
|----------|-------|
| Original images | 1,800 (300 per class) |
| Image size | 200 × 200 px (grayscale / RGB) |
| Annotation format | Pascal VOC XML → converted to YOLO `.txt` |
| Number of classes | 6 |
| Split ratio | 70 % train / 20 % val / 10 % test |

---

## Defect Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | `crazing` | Network of fine surface cracks |
| 1 | `inclusion` | Non-metallic particles trapped in the metal |
| 2 | `patches` | Irregular surface discolouration or roughness |
| 3 | `pitted_surface` | Pits formed by corrosion or mechanical damage |
| 4 | `rolled-in_scale` | Scale fragments rolled into the surface |
| 5 | `scratches` | Linear surface cuts or grooves |

---

## Folder Structure

```
dataset/
├── raw/               # Original NEU-DET images + Pascal VOC XMLs
├── yolo/              # Final YOLO-format split (training-ready)
│   ├── images/
│   │   ├── train/    # Images (orig + augmented)
│   │   ├── val/      # Images
│   │   └── test/     # Images
│   └── labels/
│       ├── train/    # .txt label files
│       ├── val/      # .txt label files
│       └── test/     # .txt label files
├── processed/         # Resized / normalized copies (640×640)
└── augmented/         # Raw augmented output (before YOLO merge)
```

---

## Data Pipeline Steps

### 1. Format Conversion (`scripts/convert_to_yolo.py`)
Converts Pascal VOC XML bounding boxes to YOLO normalized format:
```
<class_id>  <x_center>  <y_center>  <width>  <height>
```
All values are normalized to `[0.0, 1.0]` relative to image dimensions.

### 2. Stratified Train / Val / Test Split
- **70 %** → `train/` (1,260 base images per stratified class)
- **20 %** → `val/`   (360 images)
- **10 %** → `test/`  (180 images)
- Split uses `random.seed(42)` for full reproducibility.

### 3. Automated Dataset Health Monitoring (`scripts/monitor_dataset_health.py`)
Runs comprehensive validation on the dataset to detect:
- Corrupted or unreadable images.
- Duplicate images based on MD5 checksums.
- Missing labels or missing images.
- Annotation consistency including class ID bounds and coordinate validation.
- Cross-split stem duplicates (data leakage).
Outputs a health monitoring report to `reports/dataset_health/dataset_health_report.md`.

### 4. Class Balancing via Albumentations (`scripts/augment_dataset.py`)
Minority classes in the training split are augmented until every class reaches the majority count.

### 5. Parallel Preprocessing (`scripts/preprocess_pipeline.py`)
- High-speed resizing to **640 × 640** utilizing Python's `ProcessPoolExecutor`.
- Supports configurable image normalization including `min_max`, `imagenet`, and Z-score standardization.
- Saves stats to `reports/preprocessing_statistics.json` and a markdown summary to `reports/preprocessing_validation_summary.md`.

### 6. Analytics Dashboard (`scripts/generate_dataset_dashboard.py`)
- Computes overall dataset statistics.
- Computes bounding box dimensions, aspect ratio statistics, and box densities.
- Generates side-by-side distribution charts and split pie charts.
- Overlays ground-truth annotations on random samples from each split.
- Outputs files to `reports/analytics/`.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/data.yaml` | YOLOv8 dataset config (paths + class names) |
| `configs/preprocessing.yaml` | Image resizing and normalization parameters |
| `configs/augmentation.yaml` | Albumentations pipeline parameters |
| `configs/hyperparameters.yaml` | YOLOv8 training hyperparameters |
| `configs/experiment.yaml` | Experiment tracking config |

---

## Reports Generated

| Report | Location |
|--------|----------|
| Dataset Health Report | `reports/dataset_health/dataset_health_report.md` |
| Dataset Quality Inspection | `reports/quality/dataset_quality_report.md` |
| Dataset Balancing | `reports/dataset_balancing_report.md` |
| Preprocessing statistics | `reports/preprocessing_statistics.json` |
| Preprocessing validation | `reports/preprocessing_validation_summary.md` |
| Dataset Analytics Dashboard | `reports/analytics/dataset_analytics_report.md` |
| Class distribution chart | `reports/analytics/class_distribution_dashboard.png` |
| Dataset split chart | `reports/analytics/dataset_split_dashboard.png` |
| Image sample visualizations | `reports/analytics/sample_*.jpg` |
