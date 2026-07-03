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


def load_class_names():
    """Load class mapping from configuration file."""
    if CLASSES_CFG.exists():
        try:
            with open(CLASSES_CFG, "r") as f:
                data = yaml.safe_load(f)
                return {i: name for i, name in enumerate(data.get("classes", []))}
        except Exception as e:
            print(f"Warning: Failed to load classes config: {e}")
    # Fallback default NEU class names
    return {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled_in_scale",
        5: "scratches",
    }


def calculate_statistics():
    """Calculates comprehensive dataset statistics."""
    print("=" * 60)
    print("DATASET STATISTICS REPORT")
    print("=" * 60)

    class_names = load_class_names()
    splits = ["train", "val", "test"]

    total_images_all = 0
    total_annotations_all = 0
    class_counts_all = {i: 0 for i in class_names.keys()}

    print(f"{'Split':<10} | {'Images Count':<12} | {'Labels Count':<12} | {'Total Bounding Boxes':<20}")
    print("-" * 65)

    for split in splits:
        img_dir = DATASET_ROOT / "images" / split
        lbl_dir = DATASET_ROOT / "labels" / split

        img_count = 0
        lbl_count = 0
        bbox_count = 0

        if img_dir.exists():
            img_count = len([f for f in img_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
        if lbl_dir.exists():
            lbl_files = [f for f in lbl_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            lbl_count = len(lbl_files)

            # Read labels to count classes and bounding boxes
            for lbl_file in lbl_files:
                try:
                    with open(lbl_file, "r") as f:
                        lines = f.readlines()
                        bbox_count += len(lines)
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in class_counts_all:
                                    class_counts_all[class_id] += 1
                except Exception:
                    pass

        total_images_all += img_count
        total_annotations_all += lbl_count

        print(f"{split.capitalize():<10} | {img_count:<12} | {lbl_count:<12} | {bbox_count:<20}")

    print("-" * 65)
    print(f"{'Total':<10} | {total_images_all:<12} | {total_annotations_all:<12} | {sum(class_counts_all.values()):<20}")

    # Class-wise distributions
    print("\n" + "=" * 60)
    print("CLASS-WISE INSTANCE DISTRIBUTION")
    print("=" * 60)
    for class_id, name in class_names.items():
        count = class_counts_all.get(class_id, 0)
        percentage = (
            (count / sum(class_counts_all.values()) * 100)
            if sum(class_counts_all.values()) > 0
            else 0.0
        )
        print(f"Class {class_id} ({name:<16}): {count:<5} instances ({percentage:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    calculate_statistics()
