#!/usr/bin/env python3
"""
Dataset Augmentation Pipeline using Albumentations
===================================================
Real-Time Industrial Defect Detection System

Applies geometric and color augmentations to the dataset while
correctly preserving and recalculating YOLO bounding box coordinates.

Author: saniyamirjanavar-hash
Date: 2026-07-05
"""

import os
import sys
import logging
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

try:
    import cv2
    import albumentations as A
    HAS_LIBS = True
except ImportError as err:
    HAS_LIBS = False
    print(f"[WARNING] Required libraries not installed: {err}")

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "dataset_augmentation.log", mode="a"),
    ],
)
logger = logging.getLogger("dataset_augmentation")

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
YOLO_ROOT = DATASET_ROOT / "yolo"
IMAGES_DIR = YOLO_ROOT / "images"
LABELS_DIR = YOLO_ROOT / "labels"
AUGMENTED_ROOT = DATASET_ROOT / "augmented"
REPORTS_DIR = PROJECT_ROOT / "reports"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "val", "test"]
RANDOM_SEED = 42

class DatasetAugmentor:
    """Manages and runs the Albumentations-based data augmentation pipeline."""

    def __init__(
        self,
        yolo_root: Path = YOLO_ROOT,
        output_root: Path = AUGMENTED_ROOT,
        num_aug_per_img: int = 2,
    ):
        self.yolo_root = yolo_root
        self.output_root = output_root
        self.num_aug_per_img = num_aug_per_img
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Define Albumentations pipeline with YOLO bbox support
        if HAS_LIBS:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.2),
                    A.RandomResizedCrop(size=(200, 200), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_visibility=0.3,
                ),
            )
        else:
            self.transform = None

        self.stats = {
            "timestamp": datetime.now().isoformat(),
            "original_count": 0,
            "augmented_count": 0,
            "failed_count": 0,
            "split_details": {},
        }

    def _read_yolo_label(self, label_path: Path) -> tuple:
        """Read YOLO label file and return lists of bboxes and class labels."""
        bboxes = []
        class_labels = []
        if not label_path.exists():
            return bboxes, class_labels

        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: [x_center, y_center, width, height]
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Albumentations expects [x_center, y_center, width, height] in [0, 1] range
                        # Validate bounding box coordinates are strictly within bounds [0, 1]
                        bbox = [max(1e-6, min(0.999999, val)) for val in bbox]
                        
                        bboxes.append(bbox)
                        class_labels.append(class_id)
        except Exception as exc:
            logger.error(f"Error reading label file {label_path}: {exc}")

        return bboxes, class_labels

    def _write_yolo_label(self, label_path: Path, bboxes: list, class_labels: list):
        """Write YOLO bounding boxes and labels to file."""
        try:
            with open(label_path, "w") as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    bbox_str = " ".join([f"{x:.6f}" for x in bbox])
                    f.write(f"{class_id} {bbox_str}\n")
        except Exception as exc:
            logger.error(f"Error writing label file {label_path}: {exc}")

    def augment_split(self, split: str) -> dict:
        """Augment all images in a given split (train/val/test)."""
        logger.info(f"Processing split: {split}")
        img_dir = self.yolo_root / "images" / split
        lbl_dir = self.yolo_root / "labels" / split

        out_img_dir = self.output_root / "images" / split
        out_lbl_dir = self.output_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            logger.warning(f"Image directory {img_dir} does not exist.")
            return {"original": 0, "augmented": 0}

        img_files = [
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        original_count = len(img_files)
        augmented_count = 0
        failed_count = 0

        for img_path in img_files:
            # 1. Load image
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read image: {img_path.name}")
                failed_count += 1
                continue

            # 2. Load label
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            bboxes, class_labels = self._read_yolo_label(lbl_path)

            # Copy original to output directory
            shutil_dest_img = out_img_dir / img_path.name
            shutil_dest_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            cv2.imwrite(str(shutil_dest_img), img)
            self._write_yolo_label(shutil_dest_lbl, bboxes, class_labels)

            # 3. Apply augmentations multiple times
            for i in range(self.num_aug_per_img):
                try:
                    augmented = self.transform(
                        image=img, bboxes=bboxes, class_labels=class_labels
                    )
                    aug_img = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_class_labels = augmented["class_labels"]

                    # Save augmented image
                    aug_name = f"{img_path.stem}_aug_{i}"
                    aug_img_path = out_img_dir / f"{aug_name}{img_path.suffix}"
                    aug_lbl_path = out_lbl_dir / f"{aug_name}.txt"

                    cv2.imwrite(str(aug_img_path), aug_img)
                    self._write_yolo_label(aug_lbl_path, aug_bboxes, aug_class_labels)
                    augmented_count += 1
                except Exception as exc:
                    logger.debug(f"Augmentation {i} failed for {img_path.name}: {exc}")
                    failed_count += 1

        logger.info(
            f"Split {split} summary: Original={original_count}, Augmented={augmented_count}"
        )
        return {
            "original": original_count,
            "augmented": augmented_count,
            "failed": failed_count,
        }

    def generate_report(self) -> Path:
        """Generate and save Markdown augmentation report."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "augmentation_report.md"

        lines = [
            "# Dataset Augmentation Report",
            "",
            f"Generated on: {self.stats['timestamp']}",
            "",
            "## Pipeline Details",
            "",
            "- **Library**: Albumentations",
            "- **Number of augmentations per image**: " + str(self.num_aug_per_img),
            "- **Active Augmentations**:",
            "  - Horizontal Flip",
            "  - Vertical Flip",
            "  - Random Rotation (limit=90 deg)",
            "  - Random Brightness & Contrast",
            "  - Gaussian Blur",
            "  - Hue Saturation Value Shift",
            "  - Random Resized Crop",
            "  - CLAHE (Contrast Limited Adaptive Histogram Equalization)",
            "",
            "## Summary Metrics",
            "",
            f"- **Original Images Copied**: {self.stats['original_count']}",
            f"- **New Augmented Images Generated**: {self.stats['augmented_count']}",
            f"- **Failed Augmentations (e.g. bbox out of bounds)**: {self.stats['failed_count']}",
            "",
            "## Split Details",
            "",
            "| Split | Original Images | Augmented Images | Failed |",
            "|-------|-----------------|------------------|--------|",
        ]

        for split, details in self.stats["split_details"].items():
            lines.append(
                f"| {split} | {details['original']} | {details['augmented']} | {details['failed']} |"
            )

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Augmentation report written to {report_path}")
        return report_path

    def run(self):
        """Run the full augmentation pipeline."""
        if not HAS_LIBS:
            logger.error("Required libraries (OpenCV, Albumentations) missing. Aborting.")
            return

        logger.info("=" * 60)
        logger.info("STARTING DATASET AUGMENTATION PIPELINE")
        logger.info("=" * 60)

        total_orig = 0
        total_aug = 0
        total_fail = 0

        for split in SPLITS:
            res = self.augment_split(split)
            self.stats["split_details"][split] = res
            total_orig += res["original"]
            total_aug += res["augmented"]
            total_fail += res["failed"]

        self.stats["original_count"] = total_orig
        self.stats["augmented_count"] = total_aug
        self.stats["failed_count"] = total_fail

        self.generate_report()
        logger.info("=" * 60)
        logger.info("DATASET AUGMENTATION PIPELINE COMPLETED")
        logger.info(f"Total Augmented Images: {total_aug}")
        logger.info("=" * 60)


if __name__ == "__main__":
    import shutil
    # Define placeholder setup helper
    augmentor = DatasetAugmentor(num_aug_per_img=2)
    augmentor.run()
