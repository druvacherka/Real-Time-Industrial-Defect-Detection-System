#!/usr/bin/env python3
"""
Dataset Conversion to YOLOv8 Format
=====================================
Real-Time Industrial Defect Detection System

Converts the NEU Metal Surface Defects dataset into YOLOv8-compatible
directory structure with automatic data.yaml generation.

Author: saniyamirjanavar-hash
Date: 2026-07-04
"""

import os
import sys
import shutil
import random
import logging
import yaml
from pathlib import Path
from collections import defaultdict

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "convert_to_yolo.log", mode="a"),
    ],
)
logger = logging.getLogger("convert_to_yolo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "NEU-DET"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "yolo"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFECT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
RANDOM_SEED = 42


class YOLOConverter:
    """Convert raw NEU-DET dataset into YOLOv8 directory format."""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        output_dir: Path = OUTPUT_DIR,
        split_ratios: dict = None,
    ):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.split_ratios = split_ratios or SPLIT_RATIOS
        self.stats = defaultdict(lambda: defaultdict(int))

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------
    def create_yolo_directories(self):
        """Create the YOLOv8 directory tree."""
        logger.info("Creating YOLOv8 directory structure...")
        for split in self.split_ratios:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            logger.info(f"  Created {split}/ images and labels directories")

    # ------------------------------------------------------------------
    # Collect raw images
    # ------------------------------------------------------------------
    def collect_images(self) -> dict:
        """Scan raw directory and group images by class."""
        logger.info("Collecting images from raw dataset...")
        class_images = defaultdict(list)

        if not self.raw_dir.exists():
            logger.warning(f"Raw dataset directory not found: {self.raw_dir}")
            logger.info("Scanning alternative paths...")

            # Check for images already organized in dataset/raw
            alt_raw = PROJECT_ROOT / "dataset" / "raw"
            if alt_raw.exists():
                for class_dir in sorted(alt_raw.iterdir()):
                    if class_dir.is_dir() and class_dir.name in DEFECT_CLASSES:
                        imgs = [
                            f for f in class_dir.iterdir()
                            if f.suffix.lower() in SUPPORTED_EXTENSIONS
                        ]
                        class_images[class_dir.name] = imgs
                        logger.info(f"  {class_dir.name}: {len(imgs)} images")
            return dict(class_images)

        # Scan NEU-DET structure: train/images/<class>/ and validation/images/<class>/
        for subset in ["train", "validation"]:
            img_root = self.raw_dir / subset / "images"
            if not img_root.exists():
                continue
            for class_dir in sorted(img_root.iterdir()):
                if class_dir.is_dir():
                    cls_name = class_dir.name
                    imgs = [
                        f for f in class_dir.iterdir()
                        if f.suffix.lower() in SUPPORTED_EXTENSIONS
                    ]
                    class_images[cls_name].extend(imgs)
                    logger.info(
                        f"  [{subset}] {cls_name}: {len(imgs)} images"
                    )

        total = sum(len(v) for v in class_images.values())
        logger.info(f"Total images collected: {total}")
        return dict(class_images)

    # ------------------------------------------------------------------
    # Split dataset
    # ------------------------------------------------------------------
    def split_dataset(self, class_images: dict) -> dict:
        """Split images into train/val/test per class."""
        logger.info("Splitting dataset into train/val/test...")
        random.seed(RANDOM_SEED)

        splits = {s: [] for s in self.split_ratios}

        for cls_name, images in class_images.items():
            random.shuffle(images)
            n = len(images)
            train_end = int(n * self.split_ratios["train"])
            val_end = train_end + int(n * self.split_ratios["val"])

            splits["train"].extend(
                [(img, cls_name) for img in images[:train_end]]
            )
            splits["val"].extend(
                [(img, cls_name) for img in images[train_end:val_end]]
            )
            splits["test"].extend(
                [(img, cls_name) for img in images[val_end:]]
            )

        for split, items in splits.items():
            logger.info(f"  {split}: {len(items)} images")

        return splits

    # ------------------------------------------------------------------
    # Generate YOLO label
    # ------------------------------------------------------------------
    @staticmethod
    def generate_yolo_label(class_idx: int, img_width: int, img_height: int) -> str:
        """
        Generate a YOLO label line for the full image bounding box.
        Format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1].
        """
        x_center = 0.5
        y_center = 0.5
        w = 1.0
        h = 1.0
        return f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

    # ------------------------------------------------------------------
    # Copy and convert
    # ------------------------------------------------------------------
    def convert(self, splits: dict):
        """Copy images and create YOLO label files."""
        logger.info("Converting dataset to YOLO format...")

        for split, items in splits.items():
            img_out = self.output_dir / "images" / split
            lbl_out = self.output_dir / "labels" / split

            for img_path, cls_name in items:
                if cls_name in DEFECT_CLASSES:
                    cls_idx = DEFECT_CLASSES.index(cls_name)
                else:
                    logger.warning(f"  Unknown class '{cls_name}', skipping")
                    continue

                # Copy image
                dest_img = img_out / img_path.name
                try:
                    shutil.copy2(str(img_path), str(dest_img))
                except Exception as exc:
                    logger.error(f"  Failed to copy {img_path.name}: {exc}")
                    continue

                # Read dimensions for label
                img_w, img_h = 200, 200  # NEU-DET default
                if HAS_OPENCV:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img_h, img_w = img.shape[:2]

                # Write label file
                label_line = self.generate_yolo_label(cls_idx, img_w, img_h)
                lbl_file = lbl_out / f"{img_path.stem}.txt"
                lbl_file.write_text(label_line + "\n", encoding="utf-8")

                self.stats[split][cls_name] += 1

            logger.info(f"  [{split}] Converted {len(items)} image-label pairs")

    # ------------------------------------------------------------------
    # Validate pairs
    # ------------------------------------------------------------------
    def validate_pairs(self) -> bool:
        """Validate that every image has a corresponding label."""
        logger.info("Validating image-label pairs...")
        all_valid = True

        for split in self.split_ratios:
            img_dir = self.output_dir / "images" / split
            lbl_dir = self.output_dir / "labels" / split

            img_stems = {f.stem for f in img_dir.iterdir() if f.is_file()}
            lbl_stems = {f.stem for f in lbl_dir.iterdir() if f.is_file()}

            missing_lbl = img_stems - lbl_stems
            missing_img = lbl_stems - img_stems

            if missing_lbl:
                logger.warning(f"  [{split}] {len(missing_lbl)} images missing labels")
                all_valid = False
            if missing_img:
                logger.warning(f"  [{split}] {len(missing_img)} labels missing images")
                all_valid = False
            if not missing_lbl and not missing_img:
                logger.info(f"  [{split}] All pairs valid ({len(img_stems)} pairs)")

        return all_valid

    # ------------------------------------------------------------------
    # Generate data.yaml
    # ------------------------------------------------------------------
    def generate_data_yaml(self) -> Path:
        """Generate YOLOv8 data.yaml configuration file."""
        logger.info("Generating data.yaml...")

        data_config = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(DEFECT_CLASSES),
            "names": DEFECT_CLASSES,
        }

        yaml_path = CONFIGS_DIR / "data.yaml"
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as fh:
            yaml.dump(data_config, fh, default_flow_style=False, sort_keys=False)

        logger.info(f"  data.yaml saved to {yaml_path}")
        return yaml_path

    # ------------------------------------------------------------------
    # Print statistics
    # ------------------------------------------------------------------
    def print_statistics(self):
        """Display dataset conversion statistics."""
        logger.info("=" * 60)
        logger.info("CONVERSION STATISTICS")
        logger.info("=" * 60)

        grand_total = 0
        for split in self.split_ratios:
            split_total = sum(self.stats[split].values())
            grand_total += split_total
            logger.info(f"  [{split.upper()}] Total: {split_total}")
            for cls_name in DEFECT_CLASSES:
                count = self.stats[split].get(cls_name, 0)
                if count > 0:
                    logger.info(f"    {cls_name}: {count}")

        logger.info(f"  GRAND TOTAL: {grand_total}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full conversion pipeline."""
        logger.info("Starting YOLOv8 dataset conversion pipeline...")

        self.create_yolo_directories()
        class_images = self.collect_images()

        if not class_images:
            logger.warning("No images found to convert. Generating data.yaml only.")
            self.generate_data_yaml()
            return

        splits = self.split_dataset(class_images)
        self.convert(splits)
        self.validate_pairs()
        self.generate_data_yaml()
        self.print_statistics()

        logger.info("YOLOv8 conversion complete!")


def main():
    converter = YOLOConverter()
    converter.run()


if __name__ == "__main__":
    main()
