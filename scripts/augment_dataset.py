#!/usr/bin/env python3
"""
Augmentation & Class-Balancing Pipeline
=========================================
Real-Time Industrial Defect Detection System

Balances minority defect classes in the YOLO training split by generating
augmented images using Albumentations.  Configuration is loaded from
`configs/augmentation.yaml`.

Usage:
    python scripts/augment_dataset.py            # full augmentation
    python scripts/augment_dataset.py --dry-run  # preview counts only

Author: saniyamirjanavar-hash
Date:   2026-07-07  refactor: use shared config module + YAML-driven pipeline
"""

from __future__ import annotations

import sys
import random
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Allow running as a standalone script from the project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    IMAGES_DIR, LABELS_DIR, LOGS_DIR, AUG_YAML,
    SPLITS, CLASS_NAMES, NUM_CLASSES, ensure_dirs,
)

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ensure_dirs(LOGS_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "augment_dataset.log", mode="a"),
    ],
)
logger = logging.getLogger("augment_dataset")

TRAIN_IMG_DIR = IMAGES_DIR / "train"
TRAIN_LBL_DIR = LABELS_DIR / "train"


# ---------------------------------------------------------------------------
# Load augmentation config from YAML
# ---------------------------------------------------------------------------

def build_transform() -> "A.Compose":
    """Build an Albumentations Compose pipeline (from YAML or defaults)."""
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations is required: pip install albumentations")

    # Try to load from YAML
    cfg: dict = {}
    if HAS_YAML and AUG_YAML.exists():
        with open(AUG_YAML, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info(f"Loaded augmentation config from {AUG_YAML}")
    else:
        logger.warning("augmentation.yaml not found — using default pipeline")

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    sp = cfg.get("spatial", {})
    px = cfg.get("pixel", {})
    bl = cfg.get("blur", {})

    transforms = [
        A.HorizontalFlip(p=sp.get("horizontal_flip", {}).get("p", 0.5)),
        A.VerticalFlip(p=sp.get("vertical_flip", {}).get("p", 0.5)),
        A.Rotate(
            limit=sp.get("rotate", {}).get("limit", 90),
            p=sp.get("rotate", {}).get("p", 0.5),
        ),
        A.RandomBrightnessContrast(
            brightness_limit=px.get("random_brightness_contrast", {}).get("brightness_limit", 0.25),
            contrast_limit=px.get("random_brightness_contrast", {}).get("contrast_limit", 0.25),
            p=px.get("random_brightness_contrast", {}).get("p", 0.5),
        ),
        A.HueSaturationValue(p=px.get("hue_saturation_value", {}).get("p", 0.4)),
        A.CLAHE(p=px.get("clahe", {}).get("p", 0.4)),
        A.GaussianBlur(p=bl.get("gaussian_blur", {}).get("p", 0.3)),
        A.MotionBlur(p=bl.get("motion_blur", {}).get("p", 0.25)),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=cfg.get("min_visibility", 0.3),
        ),
    )

def load_yolo_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                # Ensure coordinates are within [0, 1]
                coords = [max(0.0, min(1.0, c)) for c in coords]
                # Albumentations requirements: w > 0, h > 0
                if coords[2] > 0 and coords[3] > 0:
                    boxes.append([class_id] + coords)
    return boxes

def save_yolo_labels(label_path, bboxes):
    with open(label_path, "w") as f:
        for box in bboxes:
            class_id, x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Augment & balance the NEU training split.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print class counts without generating any files.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ALBUMENTATIONS AUGMENTATION & CLASS-BALANCING PIPELINE")
    if args.dry_run:
        logger.info("  Mode: DRY RUN — no files will be written")
    logger.info("=" * 60)

    # Build transform pipeline
    transform = build_transform()

    # Analyse current training class distribution
    image_files = list(TRAIN_IMG_DIR.glob("*.jpg"))
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    image_mapped: dict[int, list[Path]] = {i: [] for i in range(NUM_CLASSES)}

    for img_path in image_files:
        lbl_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        classes_in_img: set[int] = set()
        for box in boxes:
            class_id = box[0]
            class_counts[class_id] += 1
            classes_in_img.add(class_id)
        for cid in classes_in_img:
            image_mapped[cid].append(img_path)

    logger.info("Initial training instance counts:")
    for cid, count in class_counts.items():
        logger.info(f"  {CLASS_NAMES[cid]}: {count}")

    target_count = max(class_counts.values())
    logger.info(f"Target instance count for balancing: {target_count}")

    if args.dry_run:
        logger.info("DRY RUN complete — no files written.")
        return

    # Augment minority classes
    for cid, count in class_counts.items():
        if count >= target_count:
            continue

        class_name = CLASS_NAMES[cid]
        logger.info(f"\nAugmenting minority class: {class_name} ({count} → {target_count})")

        candidates = image_mapped[cid]
        if not candidates:
            logger.warning(f"  No candidate images for {class_name}. Skipping.")
            continue

        aug_idx = 0
        current_count = count
        while current_count < target_count:
            src_img_path = random.choice(candidates)
            src_lbl_path = TRAIN_LBL_DIR / f"{src_img_path.stem}.txt"

            image = cv2.imread(str(src_img_path))
            if image is None:
                continue

            yolo_boxes = load_yolo_labels(src_lbl_path)
            if not yolo_boxes:
                continue

            bboxes = [box[1:] for box in yolo_boxes]
            class_labels = [box[0] for box in yolo_boxes]

            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["class_labels"]
            except Exception:
                continue

            if not aug_bboxes:
                continue

            dst_stem = f"aug_{src_img_path.stem}_{aug_idx}"
            dst_img_path = TRAIN_IMG_DIR / f"{dst_stem}.jpg"
            dst_lbl_path = TRAIN_LBL_DIR / f"{dst_stem}.txt"

            cv2.imwrite(str(dst_img_path), aug_img)

            assembled: list[list] = []
            for lbl, bbox in zip(aug_labels, aug_bboxes):
                assembled.append([lbl, *bbox])
                if lbl == cid:
                    current_count += 1

            save_yolo_labels(dst_lbl_path, assembled)
            aug_idx += 1

        logger.info(f"  Generated {aug_idx} augmented images for {class_name}. "
                    f"Final count: {current_count}")

    logger.info("\nAugmentation and class balancing completed.")


if __name__ == "__main__":
    main()
