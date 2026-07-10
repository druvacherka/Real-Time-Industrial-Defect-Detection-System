# Dataset Quality Inspection & Integrity Report

This report provides an automated validation summary of the NEU Metal Surface Defects dataset.

## 1. Directory Structure Health Check

| Directory | Status |
| --- | --- |
| `dataset` | ✅ EXISTS |
| `dataset\yolo` | ✅ EXISTS |
| `dataset\yolo\images` | ✅ EXISTS |
| `dataset\yolo\labels` | ✅ EXISTS |
| `dataset\yolo\images\train` | ✅ EXISTS |
| `dataset\yolo\labels\train` | ✅ EXISTS |
| `dataset\yolo\images\val` | ✅ EXISTS |
| `dataset\yolo\labels\val` | ✅ EXISTS |
| `dataset\yolo\images\test` | ✅ EXISTS |
| `dataset\yolo\labels\test` | ✅ EXISTS |

## 2. Image-Label Pair Consistency

- **Missing Label Files (Images without labels)**: 0
- **Missing Image Files (Labels without images)**: 0
- **Empty Annotation Files**: 0

## 3. Image Corruption Analysis

- **Corrupted/Unreadable Images**: 0

## 4. Image Duplication and Leakage Check

- **Duplicate image groups (Same MD5 within or across splits)**: 2
  - **Group 1** (MD5: `62a4ea01743534d1a9e321e6e7d0ef45`):
    - `train/aug_pitted_surface_102_68.jpg`
    - `train/aug_pitted_surface_102_74.jpg`
  - **Group 2** (MD5: `24f39e0f91f95e4c785052ac4d3e49a6`):
    - `train/patches_101.jpg`
    - `val/patches_105.jpg`

## 5. Cross-Split Stem Duplicates (Potential Data Leakage)

- **Stems duplicated across splits**: 0

## 6. Bounding Box & Annotation Consistency

- **Total bounding boxes checked**: 5852
- **Total consistency/class errors**: 0

### Class Distribution (Valid annotations)

| Class ID | Class Name | Annotations |
| --- | --- | --- |
| 0 | crazing | 976 |
| 1 | inclusion | 1083 |
| 2 | patches | 1017 |
| 3 | pitted_surface | 880 |
| 4 | rolled-in_scale | 965 |
| 5 | scratches | 931 |

---
Report generated automatically by `inspect_dataset_quality.py`.