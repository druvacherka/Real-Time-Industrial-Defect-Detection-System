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

        # 1. Check folder existence and emptiness
        for folder in [img_split_dir, lbl_split_dir]:
            if not folder.exists():
                print(f"  [MISSING] Folder does not exist: {folder}")
                has_issues = True
                continue

            # Exclude .gitkeep or other dotfiles when checking empty folders
            visible_files = [f for f in folder.iterdir() if not f.name.startswith(".")]
            if len(visible_files) == 0:
                print(f"  [EMPTY] Folder is empty: {folder}")

        # 2. Match images and labels if directories exist
        if img_split_dir.exists() and lbl_split_dir.exists():
            img_files = {f.stem: f for f in img_split_dir.iterdir() if f.is_file() and not f.name.startswith(".")}
            lbl_files = {f.stem: f for f in lbl_split_dir.iterdir() if f.is_file() and not f.name.startswith(".")}

            # Check for images without corresponding labels
            missing_labels = img_files.keys() - lbl_files.keys()
            if missing_labels:
                print(f"  [WARNING] {len(missing_labels)} images have no corresponding label file:")
                for stem in sorted(list(missing_labels))[:5]:
                    print(f"    - {img_files[stem].name} is missing label file")
                if len(missing_labels) > 5:
                    print(f"    - ... and {len(missing_labels) - 5} more")
                has_issues = True

            # Check for labels without corresponding images
            missing_images = lbl_files.keys() - img_files.keys()
            if missing_images:
                print(f"  [WARNING] {len(missing_images)} label files have no corresponding image:")
                for stem in sorted(list(missing_images))[:5]:
                    print(f"    - {lbl_files[stem].name} is missing image file")
                if len(missing_images) > 5:
                    print(f"    - ... and {len(missing_images) - 5} more")
                has_issues = True

            if not missing_labels and not missing_images:
                print(f"  [OK] All images and label files match correctly (Count: {len(img_files)}).")

    print("\n" + "=" * 60)
    if has_issues:
        print("Verification completed with warnings/errors. Please inspect logs.")
    else:
        print("Verification completed successfully. No issues detected.")
    print("=" * 60)
    return not has_issues


if __name__ == "__main__":
    verify_dataset()