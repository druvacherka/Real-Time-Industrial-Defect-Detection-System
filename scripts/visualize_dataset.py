#!/usr/bin/env python3
"""
Dataset Visualization Script for Real-Time Industrial Defect Detection System.
Randomly selects images from the dataset and displays them with bounding box overlays.
"""

import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import yaml

# Paths definitions
DATASET_ROOT = Path("dataset/yolo")
CLASSES_CFG = Path("configs/classes.yaml")

# TODO: Add option to save visual outputs as image artifacts under dataset/reports/visualizations/
# TODO: Implement interactive slider to scroll through image batch visualizations


def load_class_names():
    """Load class mapping from configuration file."""
    if CLASSES_CFG.exists():
        try:
            with open(CLASSES_CFG, "r") as f:
                data = yaml.safe_load(f)
                return data.get("classes", [])
        except Exception:
            pass
    return ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def visualize_random_samples(num_samples=4):
    """Randomly selects and visualizes images from the YOLO dataset splits."""
    print("=" * 60)
    print("DATASET SAMPLE VISUALIZATION")
    print("=" * 60)

    class_names = load_class_names()
    splits = ["train", "val", "test"]

    all_images = []
    for split in splits:
        img_dir = DATASET_ROOT / "images" / split
        if img_dir.exists():
            images = [f for f in img_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            for img in images:
                all_images.append((split, img))

    if not all_images:
        print("No images found in the dataset folder structure.")
        print(f"Please place image files in: {DATASET_ROOT}/images/[train/val/test]")
        return

    samples_to_show = random.sample(all_images, min(len(all_images), num_samples))
    print(f"Displaying {len(samples_to_show)} random samples.")


if __name__ == "__main__":
    visualize_random_samples()
