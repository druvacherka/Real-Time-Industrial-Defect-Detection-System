# Real-Time Industrial Defect Detection System

A real-time computer vision system for detecting industrial surface defects using **YOLOv8**.

## Dataset Pipeline Overview

This repository initializes a standardized dataset pipeline to preprocess, verify, and format the Northeastern University (NEU) Metal Surface Defects Dataset.

### Dataset Directory Structure
We organize the data into separate directories for raw input, processed intermediates, YOLO formatting, and reporting:
* `dataset/raw/`: Original NEU dataset images and annotations.
* `dataset/processed/`: Standardized dataset formatting output.
* `dataset/yolo/`: YOLOv8 training/validation splits.
* `dataset/annotations/`: Converted annotations.
* `dataset/augmented/`: Augmentation pipeline results.
* `dataset/reports/`: Automation reports and metrics.

### Dataset Workflow Usage

To manage and verify the dataset setup, the following utility scripts are available:

#### 1. Dataset Verification
Run the following script to check the integrity of YOLO splits (missing images/labels, empty directories):
```bash
python scripts/verify_dataset.py
```

#### 2. Dataset Statistics
Run this script to calculate the distribution of images, labels, and class instances across the splits:
```bash
python scripts/dataset_statistics.py
```

#### 3. Dataset Visualization
Run this script to display random dataset samples complete with bounding box overlays:
```bash
python scripts/visualize_dataset.py
```