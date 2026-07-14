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

### 3. Advanced Dataset Validation & Integrity Checks (`scripts/verify_dataset.py`)
Runs comprehensive validation on the dataset to verify structure, image-label mapping, and annotation integrity:
- Checks folder existence for images/labels split directories.
- Identifies missing labels (images without labels) and missing images (labels without images).
- Scans for corrupted/unreadable image files.
- Computes MD5 checksums to detect duplicate images.
- Checks for duplicate stems across splits to detect data leakage.
- Validates annotation consistency: checks number of tokens (exactly 5), numeric parse correctness, coordinate values (between 0.0 and 1.0).
- **Advanced box checks**: detects overlapping bounding boxes (IoU > 0.90) in the same image, and flags extremely small boxes (relative width/height < 0.005).
- Outputs a validation report to `reports/dataset_validation_report.md` and a comprehensive data integrity report to `reports/dataset_integrity_report.md`.

### 4. Class Balancing via Albumentations (`scripts/augment_dataset.py`)
Minority classes in the training split are augmented until every class reaches the majority class count (or a specified target count). 
- The augmentation pipeline is optimized to use 9 distinct Albumentations transforms: HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast, GaussianBlur, MotionBlur, CLAHE, and HueSaturationValue.
- Ensures bounding boxes are safe post-augmentation (drops boxes falling below 30% visibility).
- Generates a balancing report in `reports/dataset_balancing_report.md` and charts in `reports/graphs/`.

### 5. Parallel Preprocessing (`scripts/preprocess_pipeline.py`)
- High-speed resizing to **640 × 640** utilizing Python's `ProcessPoolExecutor` for concurrency.
- Supports configurable image normalization including `min_max`, `imagenet`, and Z-score standardization.
- **Detailed Preprocessing Metrics**: collects timing breakdown across four stages:
  1. *Read & Validate*
  2. *Resize*
  3. *Normalization*
  4. *Save & Format*
- Saves metrics breakdown to `reports/preprocessing_statistics.json` and a markdown summary to `reports/preprocessing_validation_summary.md`.

### 6. Analytics Dashboard (`scripts/generate_dataset_dashboard.py`)
- Computes overall dataset statistics.
- Computes bounding box dimensions, aspect ratio statistics, and box densities.
- Generates representation charts and split pie charts.
- Overlays ground-truth annotations on random samples from each split using premium alpha transparency blending (fill factor 25%).
- Outputs files to `reports/analytics/`.

### 7. Dataset Versioning and Metadata Management (`utils/metadata_manager.py` / `scripts/manage_metadata.py`)
- Automatic management and tracking of semantic dataset version numbers (e.g. `v1.0.0`, `v1.0.1`, etc.).
- Compiles metadata metrics (total image counts, bounding boxes per split, total file size, and class distributions).
- Saves metadata database files directly as JSON to `dataset/metadata.json` for integration support.

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
| Dataset Validation Report | `reports/dataset_validation_report.md` |
| Dataset Integrity Report | `reports/dataset_integrity_report.md` |
| Dataset Quality Inspection | `reports/quality/dataset_quality_report.md` |
| Dataset Balancing | `reports/dataset_balancing_report.md` |
| Preprocessing statistics | `reports/preprocessing_statistics.json` |
| Preprocessing validation | `reports/preprocessing_validation_summary.md` |
| Dataset Analytics Dashboard | `reports/analytics/dataset_analytics_report.md` |
| Dataset Metadata Database | `dataset/metadata.json` |
| Class distribution chart | `reports/analytics/class_distribution_dashboard.png` |
| Dataset split chart | `reports/analytics/dataset_split_dashboard.png` |
| Image sample visualizations | `reports/analytics/sample_*.jpg` |
