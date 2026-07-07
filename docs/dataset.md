# NEU Metal Surface Defects — Dataset Documentation

> **Last updated:** 2026-07-07  
> **Author:** saniyamirjanavar-hash

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
│   │   ├── train/    # 1,931 images (orig + augmented)
│   │   ├── val/      #   361 images
│   │   └── test/     #   181 images
│   └── labels/
│       ├── train/    # 1,931 .txt label files
│       ├── val/      #   361 .txt label files
│       └── test/     #   181 .txt label files
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

### 3. Class Balancing via Albumentations (`scripts/augment_dataset.py`)
Minority classes in the training split are augmented until every class reaches the majority count. Transforms applied:

| Category | Transform | Probability |
|----------|-----------|-------------|
| Spatial | Horizontal Flip | 0.5 |
| Spatial | Vertical Flip | 0.5 |
| Spatial | Rotate (±90°) | 0.5 |
| Spatial | Shift-Scale-Rotate | 0.4 |
| Pixel | Random Brightness + Contrast | 0.5 |
| Pixel | Hue-Saturation-Value | 0.4 |
| Pixel | CLAHE | 0.4 |
| Blur | Gaussian Blur | 0.3 |
| Blur | Motion Blur | 0.25 |

Augmented files are named `aug_<original_stem>_<idx>.jpg`.

### 4. Preprocessing (`scripts/preprocess_pipeline.py`)
- Resize to **640 × 640** using `INTER_AREA` (downscale) / `INTER_LINEAR` (upscale)
- Optional pixel normalization to `[0.0, 1.0]`
- Corrupted image detection and removal

### 5. Validation (`scripts/validate_pairs.py`)
- Image readability check (OpenCV)
- YOLO format: 5 fields per annotation line
- Class ID in `[0, 5]`
- Bbox coordinates in `[0.0, 1.0]`
- Positive width and height
- Orphan image / label detection

### 6. Preprocessing Report (`scripts/optimize_preprocess.py`)
- Duplicate image detection (MD5 hash)
- Float class ID normalization: `0.0` → `0`
- Outputs `reports/preprocessing_validation_summary.md`

---

## Final Dataset Counts

| Split | Images | Labels |
|-------|--------|--------|
| Train | 1,931  | 1,931  |
| Val   |   361  |   361  |
| Test  |   181  |   181  |
| **Total** | **2,473** | **2,473** |

> Train includes augmented samples (`aug_*` prefix) to balance minority classes.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/data.yaml` | YOLOv8 dataset config (paths + class names) |
| `configs/augmentation.yaml` | Albumentations pipeline parameters |
| `configs/hyperparameters.yaml` | YOLOv8 training hyperparameters |
| `configs/experiment.yaml` | Experiment tracking config |

---

## Running the Pipeline

```bash
# 1. Convert raw XML → YOLO (first-time only)
python scripts/convert_to_yolo.py

# 2. Full dataset build (fix IDs + validate + write data.yaml + summary)
python scripts/build_final_dataset.py

# 3. Augment & balance (if re-running augmentation)
python scripts/augment_dataset.py

# 4. Validate all pairs
python scripts/validate_pairs.py

# 5. Generate statistics + charts
python scripts/dataset_stats.py

# 6. Generate quality reports
python scripts/generate_reports.py
```

---

## Reports Generated

| Report | Location |
|--------|----------|
| Dataset quality | `reports/dataset_quality_report.md` |
| Class distribution | `reports/class_distribution.md` |
| Preprocessing validation | `reports/preprocessing_validation_summary.md` |
| Pair validation | `reports/validation_report.md` |
| Final summary | `reports/final_dataset_summary.md` |
| Bar chart | `reports/graphs/class_distribution_bar.png` |
| Pie chart | `reports/graphs/class_distribution_pie.png` |
| Per-split chart | `reports/graphs/class_distribution_per_split.png` |
