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
