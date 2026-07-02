#!/usr/bin/env python3
"""
Dataset Verification Script for Real-Time Industrial Defect Detection System.
Checks for missing images, missing label files, and empty directories.
"""

import sys
from pathlib import Path

# Paths definitions
DATASET_ROOT = Path("dataset/yolo")
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

# TODO: Integrate this verification check as a pre-commit hook or CI/CD pipeline step
# TODO: Add automatic corrupted image detection using PIL/OpenCV image headers verification


def verify_dataset():
    """Performs integrity checks on the dataset splits."""
    print("=" * 60)
    print("DATASET INTEGRITY VERIFICATION")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        print(f"Error: Dataset directory '{DATASET_ROOT}' does not exist.")
        print("Please initialize dataset directories first.")
        return False

    splits = ["train", "val", "test"]
    has_issues = False

    for split in splits:
        print(f"\nChecking split: [{split.upper()}]")
        img_split_dir = IMAGES_DIR / split
        lbl_split_dir = LABELS_DIR / split

        for folder in [img_split_dir, lbl_split_dir]:
            if not folder.exists():
                print(f"  [MISSING] Folder does not exist: {folder}")
                has_issues = True
                continue

            visible_files = [f for f in folder.iterdir() if not f.name.startswith(".")]
            if len(visible_files) == 0:
                print(f"  [EMPTY] Folder is empty: {folder}")

        if img_split_dir.exists() and lbl_split_dir.exists():
            img_files = {f.stem: f for f in img_split_dir.iterdir() if f.is_file() and not f.name.startswith(".")}
            lbl_files = {f.stem: f for f in lbl_split_dir.iterdir() if f.is_file() and not f.name.startswith(".")}

            missing_labels = img_files.keys() - lbl_files.keys()
            if missing_labels:
                print(f"  [WARNING] {len(missing_labels)} images have no corresponding label file.")
                has_issues = True

            missing_images = lbl_files.keys() - img_files.keys()
            if missing_images:
                print(f"  [WARNING] {len(missing_images)} label files have no corresponding image.")
                has_issues = True

            if not missing_labels and not missing_images:
                print(f"  [OK] All images and label files match correctly.")

    print("\n" + "=" * 60)
    return not has_issues


if __name__ == "__main__":
    verify_dataset()