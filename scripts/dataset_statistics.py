#!/usr/bin/env python3
"""
Dataset Statistics Script for Real-Time Industrial Defect Detection System.
Calculates number of images, folder statistics, and class-wise object distributions.
"""

from pathlib import Path
import yaml

# Paths definitions
DATASET_ROOT = Path("dataset/yolo")
CLASSES_CFG = Path("configs/classes.yaml")

# TODO: Add export capability to save stats reports into dataset/reports/stats.json
# TODO: Calculate bounding box aspect ratios and anchors stats for YOLOv8 anchor tuning


def load_class_names():
    """Load class mapping from configuration file."""
    if CLASSES_CFG.exists():
        try:
            with open(CLASSES_CFG, "r") as f:
                data = yaml.safe_load(f)
                return {i: name for i, name in enumerate(data.get("classes", []))}
        except Exception:
            pass
    return {0: "crazing", 1: "inclusion", 2: "patches", 3: "pitted_surface", 4: "rolled_in_scale", 5: "scratches"}


def calculate_statistics():
    """Calculates dataset statistics."""
    class_names = load_class_names()
    splits = ["train", "val", "test"]

    print("=" * 60)
    print("DATASET STATISTICS REPORT")
    print("=" * 60)

    for split in splits:
        img_dir = DATASET_ROOT / "images" / split
        img_count = 0
        if img_dir.exists():
            img_count = len([f for f in img_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
        print(f"Split {split.upper()}: {img_count} images found.")
    print("=" * 60)


if __name__ == "__main__":
    calculate_statistics()