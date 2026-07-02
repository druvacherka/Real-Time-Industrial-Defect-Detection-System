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