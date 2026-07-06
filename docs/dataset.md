# NEU Metal Surface Defects Dataset Pipeline

This document provides a detailed overview of the dataset pipeline for training a YOLOv8 object detection model on the NEU Metal Surface Defects Dataset.

## Dataset Overview
The NEU (Northeastern University) Metal Surface Defects Dataset contains six kinds of typical surface defects of hot-rolled steel strip:
1. **Crazing (Cr)**
2. **Inclusion (In)**
3. **Patches (Pa)**
4. **Pitted Surface (Ps)**
5. **Rolled-in Scale (Rs)**
6. **Scratches (Sc)**

Each class consists of 300 samples of 200×200 pixel grayscale/RGB images, for a total of 1,800 images.

## Folder Structure
The pipeline organizes the data as follows:
- `dataset/raw/`: Original unaltered images and XML annotation files.
- `dataset/processed/`: Standardized dataset versions (e.g., resizing, formatting).
- `dataset/yolo/`: Organized splits for YOLO training.
  - `images/`: Subdivided into `train/`, `val/`, and `test/` images.
  - `labels/`: Subdivided into `train/`, `val/`, and `test/` annotation labels (`.txt` files containing bounding boxes).
- `dataset/annotations/`: Intermediate or formatted annotations (e.g., converted XML to JSON/CSV).
- `dataset/augmented/`: Output directory for augmented samples created during offline training pipelines.
- `dataset/reports/`: Data analysis, stats, verification logs, and visualization reports.

## YOLO Format Explanation
YOLO format annotations require a separate text file (`.txt`) for each image. If an image contains no objects, no `.txt` file is required, though a blank one is acceptable.
Each line in the `.txt` file represents a single object bounding box and uses the following format:
```text
<class_id> <x_center> <y_center> <width> <height>
```
Where:
- `class_id`: Integer class index (from 0 to 5 for NEU dataset).
- `x_center`, `y_center`: Bounding box center coordinates, normalized by the image's width and height (values range from 0.0 to 1.0).
- `width`, `height`: Bounding box dimensions, normalized by the image's width and height (values range from 0.0 to 1.0).

## Completed Preprocessing & Augmentation Pipeline
To prepare raw NEU-DET data (which uses Pascal VOC XML annotations) for training, the pipeline executes the following steps:
1. **Verification**: Detects corrupted files, checks for missing annotations, and flags duplicate image sets.
2. **Format Conversion**: Converts XML coordinates (`xmin`, `ymin`, `xmax`, `ymax`) to YOLO normalized center-based coordinates (`x_center`, `y_center`, `width`, `height`).
3. **Train-Val-Test Splitting**: Partition the 1,800 images into:
   - 70% Train (1,260 images)
   - 20% Validation (360 images)
   - 10% Test (180 images)
   Using a stratified split to maintain class balance across sets.
4. **Albumentations Augmentation & Class Balancing**:
   - Applies an offline augmentation pipeline exclusively to minority class instances in the training split.
   - Transformations include spatial modifications (Horizontal Flip, Vertical Flip, Rotation) and pixel enhancements (CLAHE, Random Brightness & Contrast, Hue Saturation Value, Gaussian Blur, Motion Blur).
   - Dynamically balances all class instances to reach the majority count (~690 instances).
5. **Statistical Verification & Reports**: Saves class distribution charts and writes quality/preprocessing summaries in `dataset/reports/`.
