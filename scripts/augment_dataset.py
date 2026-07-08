#!/usr/bin/env python3
"""
Augmentation & Class-Balancing Pipeline  (v2)
==============================================
Real-Time Industrial Defect Detection System

Balances minority defect classes in the YOLO training split by generating
augmented images via an Albumentations pipeline configured from
``configs/augmentation.yaml``.

Pipeline overview
-----------------
1. Load augmentation config from YAML (falls back to hardcoded defaults).
2. Build an Albumentations Compose pipeline with all 8 transforms:
     HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
     HueSaturationValue, CLAHE, GaussianBlur, MotionBlur.
3. Analyse the *before* class distribution.
4. Augment every minority class up to the majority-class target count.
5. Save augmented images + YOLO labels to the train split (or augmented/).
6. Compute *after* distribution.
7. Save augmentation statistics to ``reports/augmentation_stats.json``.
8. Generate before/after comparison charts (PNG).

Usage:
    python scripts/augment_dataset.py              # full run
    python scripts/augment_dataset.py --dry-run    # preview counts only
    python scripts/augment_dataset.py --no-charts  # skip chart generation

Author: saniyamirjanavar-hash
Date:   2026-07-08  feat(data): optimize augmentation pipeline and
                    dataset balancing
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: allow running directly from the project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    IMAGES_DIR,
    LABELS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
    LOGS_DIR,
    AUG_YAML,
    SPLITS,
    CLASS_NAMES,
    NUM_CLASSES,
    DEFECT_CLASSES,
    CLASS_COLORS,
    ensure_dirs,
    iter_images,
    iter_labels,
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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

# Convenience paths
TRAIN_IMG_DIR: Path = IMAGES_DIR / "train"
TRAIN_LBL_DIR: Path = LABELS_DIR / "train"


# ===========================================================================
# Config loading
# ===========================================================================

def load_aug_config() -> dict:
    """
    Load augmentation config from ``configs/augmentation.yaml``.

    Returns the parsed dict, or an empty dict if the file is absent or
    ``PyYAML`` is not installed (callers rely on .get() with defaults).
    """
    if not HAS_YAML:
        logger.warning("PyYAML not installed — using hardcoded augmentation defaults")
        return {}
    if not AUG_YAML.exists():
        logger.warning("augmentation.yaml not found at %s — using defaults", AUG_YAML)
        return {}
    with open(AUG_YAML, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    logger.info("Loaded augmentation config from %s", AUG_YAML)
    return cfg


# ===========================================================================
# Transform builder
# ===========================================================================

def build_transform(cfg: dict) -> "A.Compose":
    """
    Build the full Albumentations pipeline from ``cfg``.

    Includes all 8 required transforms:
        HorizontalFlip, VerticalFlip, Rotate,
        RandomBrightnessContrast, HueSaturationValue, CLAHE,
        GaussianBlur, MotionBlur.

    Args:
        cfg: Parsed augmentation.yaml content.

    Returns:
        An ``A.Compose`` object with bbox support.

    Raises:
        RuntimeError: If albumentations is not installed.
    """
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError(
            "albumentations is required: pip install albumentations"
        )

    sp  = cfg.get("spatial", {})
    px  = cfg.get("pixel", {})
    bl  = cfg.get("blur", {})

    # ── Spatial transforms ───────────────────────────────────────────────
    horizontal_flip = A.HorizontalFlip(
        p=sp.get("horizontal_flip", {}).get("p", 0.5)
    )

    vertical_flip = A.VerticalFlip(
        p=sp.get("vertical_flip", {}).get("p", 0.5)
    )

    rotate = A.Rotate(
        limit=sp.get("rotate", {}).get("limit", 90),
        border_mode=sp.get("rotate", {}).get("border_mode", 0),
        p=sp.get("rotate", {}).get("p", 0.5),
    )

    shift_scale_rotate = A.ShiftScaleRotate(
        shift_limit=sp.get("shift_scale_rotate", {}).get("shift_limit", 0.0625),
        scale_limit=sp.get("shift_scale_rotate", {}).get("scale_limit", 0.10),
        rotate_limit=sp.get("shift_scale_rotate", {}).get("rotate_limit", 45),
        border_mode=sp.get("shift_scale_rotate", {}).get("border_mode", 0),
        p=sp.get("shift_scale_rotate", {}).get("p", 0.4),
    )

    # ── Pixel / colour transforms ────────────────────────────────────────
    random_brightness_contrast = A.RandomBrightnessContrast(
        brightness_limit=px.get("random_brightness_contrast", {}).get(
            "brightness_limit", 0.30
        ),
        contrast_limit=px.get("random_brightness_contrast", {}).get(
            "contrast_limit", 0.30
        ),
        p=px.get("random_brightness_contrast", {}).get("p", 0.5),
    )

    hue_sat_val = A.HueSaturationValue(
        hue_shift_limit=px.get("hue_saturation_value", {}).get("hue_shift_limit", 15),
        sat_shift_limit=px.get("hue_saturation_value", {}).get("sat_shift_limit", 30),
        val_shift_limit=px.get("hue_saturation_value", {}).get("val_shift_limit", 20),
        p=px.get("hue_saturation_value", {}).get("p", 0.4),
    )

    clahe = A.CLAHE(
        clip_limit=px.get("clahe", {}).get("clip_limit", 4.0),
        tile_grid_size=tuple(
            px.get("clahe", {}).get("tile_grid_size", [8, 8])
        ),
        p=px.get("clahe", {}).get("p", 0.4),
    )

    # ── Blur & noise transforms ──────────────────────────────────────────
    blur_limit_raw = bl.get("gaussian_blur", {}).get("blur_limit", [3, 7])
    if isinstance(blur_limit_raw, int):
        blur_limit_raw = [3, blur_limit_raw]
    gaussian_blur = A.GaussianBlur(
        blur_limit=tuple(blur_limit_raw),
        p=bl.get("gaussian_blur", {}).get("p", 0.30),
    )

    motion_blur = A.MotionBlur(
        blur_limit=bl.get("motion_blur", {}).get("blur_limit", 7),
        p=bl.get("motion_blur", {}).get("p", 0.25),
    )

    transforms = [
        horizontal_flip,
        vertical_flip,
        rotate,
        shift_scale_rotate,
        random_brightness_contrast,
        hue_sat_val,
        clahe,
        gaussian_blur,
        motion_blur,
    ]

    min_vis = cfg.get("min_visibility", 0.3)

    logger.info(
        "Built Albumentations pipeline with %d transforms (min_visibility=%.2f)",
        len(transforms), min_vis,
    )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=min_vis,
        ),
    )


# ===========================================================================
# YOLO label I/O helpers
# ===========================================================================

def load_yolo_labels(label_path: Path) -> list[list]:
    """
    Read a YOLO label file and return validated bounding boxes.

    Returns:
        List of [class_id, cx, cy, w, h] entries.  Entries with
        zero-size bboxes or out-of-range coordinates are silently dropped.
    """
    boxes: list[list] = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            # Clamp to [0, 1]
            coords = [max(0.0, min(1.0, c)) for c in coords]
            # Require strictly positive width and height
            if coords[2] > 0.0 and coords[3] > 0.0:
                boxes.append([cls_id] + coords)
    return boxes


def save_yolo_labels(label_path: Path, bboxes: list[list]) -> None:
    """Write YOLO-format bounding boxes to a .txt label file."""
    with open(label_path, "w", encoding="utf-8") as fh:
        for box in bboxes:
            cls_id, cx, cy, bw, bh = box
            fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ===========================================================================
# Distribution helpers
# ===========================================================================

def count_class_distribution(
    img_dir: Path, lbl_dir: Path
) -> tuple[dict[int, int], dict[int, list[Path]]]:
    """
    Count per-class annotation instances and which images contain each class.

    Args:
        img_dir: Directory with .jpg images.
        lbl_dir: Directory with matching .txt labels.

    Returns:
        ``(class_counts, image_map)`` where
        ``class_counts[cls_id]`` is the annotation count and
        ``image_map[cls_id]`` is a list of image Paths containing that class.
    """
    class_counts: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    image_map: dict[int, list[Path]] = {i: [] for i in range(NUM_CLASSES)}

    for img_path in img_dir.glob("*.jpg"):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        classes_in_img: set[int] = set()
        for box in boxes:
            cls_id = box[0]
            if 0 <= cls_id < NUM_CLASSES:
                class_counts[cls_id] += 1
                classes_in_img.add(cls_id)
        for cid in classes_in_img:
            image_map[cid].append(img_path)

    return class_counts, image_map


# ===========================================================================
# Chart generation
# ===========================================================================

def generate_distribution_charts(
    before: dict[int, int],
    after: dict[int, int],
    charts_dir: Path,
) -> None:
    """
    Save before/after comparison charts for the augmentation pipeline.

    Generates:
        - augmentation_before_after_bar.png  — side-by-side bar chart
        - augmentation_comparison_pie.png    — two pie charts

    Args:
        before: Per-class counts before augmentation.
        after:  Per-class counts after augmentation.
        charts_dir: Output directory for PNG files.
    """
    if not HAS_MPL:
        logger.warning("Matplotlib not available — skipping chart generation")
        return

    ensure_dirs(charts_dir)
    class_labels = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    before_counts = [before.get(i, 0) for i in range(NUM_CLASSES)]
    after_counts  = [after.get(i, 0)  for i in range(NUM_CLASSES)]
    x = np.arange(NUM_CLASSES)
    width = 0.35

    # ── Before/after bar chart ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_before = ax.bar(
        x - width / 2, before_counts, width,
        label="Before Augmentation", color="#4C72B0", edgecolor="white",
    )
    bars_after = ax.bar(
        x + width / 2, after_counts, width,
        label="After Augmentation", color="#DD8452", edgecolor="white",
    )

    # Value labels
    for bar in bars_before:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                str(int(h)), ha="center", va="bottom", fontsize=8,
            )
    for bar in bars_after:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.5,
                str(int(h)), ha="center", va="bottom", fontsize=8,
                color="#DD5500",
            )

    ax.set_xlabel("Defect Class", fontsize=12)
    ax.set_ylabel("Annotation Count", fontsize=12)
    ax.set_title(
        "Class Distribution — Before vs After Augmentation",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=20, ha="right")
    ax.legend(fontsize=11)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    bar_path = charts_dir / "augmentation_before_after_bar.png"
    fig.savefig(str(bar_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Before/after bar chart saved: %s", bar_path)

    # ── Side-by-side pie charts ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, counts, title in [
        (axes[0], before_counts, "Before Augmentation"),
        (axes[1], after_counts,  "After Augmentation"),
    ]:
        if sum(counts) > 0:
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=class_labels,
                autopct="%1.1f%%",
                colors=CLASS_COLORS[:NUM_CLASSES],
                startangle=140,
                textprops={"fontsize": 9},
            )
            for at in autotexts:
                at.set_fontweight("bold")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle(
        "Defect Class Distribution — Augmentation Comparison",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    pie_path = charts_dir / "augmentation_comparison_pie.png"
    fig.savefig(str(pie_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Comparison pie chart saved: %s", pie_path)


# ===========================================================================
# Augmentation statistics
# ===========================================================================

def save_augmentation_stats(
    before: dict[int, int],
    after: dict[int, int],
    generated_per_class: dict[int, int],
    elapsed_sec: float,
    output_path: Path,
) -> None:
    """
    Persist a JSON file with detailed augmentation statistics.

    Args:
        before: Per-class counts before augmentation.
        after:  Per-class counts after augmentation.
        generated_per_class: Number of new images generated per class.
        elapsed_sec: Total augmentation wall-clock time in seconds.
        output_path: Where to save the .json file.
    """
    ensure_dirs(output_path.parent)

    stats: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed_sec, 2),
        "per_class": {},
        "totals": {
            "before_total": sum(before.values()),
            "after_total": sum(after.values()),
            "total_generated": sum(generated_per_class.values()),
        },
    }

    for cls_id in range(NUM_CLASSES):
        cls_name = DEFECT_CLASSES.get(cls_id, f"unknown_{cls_id}")
        b = before.get(cls_id, 0)
        a = after.get(cls_id, 0)
        gen = generated_per_class.get(cls_id, 0)
        delta = a - b
        pct_increase = round((delta / b * 100) if b > 0 else 0.0, 2)

        stats["per_class"][cls_name] = {
            "class_id": cls_id,
            "before": b,
            "after": a,
            "generated_images": gen,
            "delta": delta,
            "pct_increase": pct_increase,
        }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("  Augmentation stats saved: %s", output_path)


# ===========================================================================
# Main augmentation pipeline
# ===========================================================================

def run_augmentation(cfg: dict, dry_run: bool, generate_charts: bool) -> None:
    """
    Execute the full augmentation and class-balancing pipeline.

    Args:
        cfg:             Parsed augmentation YAML config.
        dry_run:         If True, only log what would happen — no files written.
        generate_charts: If True, save before/after distribution charts.
    """
    logger.info("=" * 60)
    logger.info("AUGMENTATION & CLASS-BALANCING PIPELINE (v2)")
    if dry_run:
        logger.info("  Mode: DRY RUN — no files will be written")
    logger.info("=" * 60)

    # 1. Build transform pipeline
    transform = build_transform(cfg)

    # 2. Analyse BEFORE distribution
    if not TRAIN_IMG_DIR.exists():
        logger.error(
            "Train image directory not found: %s", TRAIN_IMG_DIR
        )
        return

    logger.info("Analysing initial class distribution (BEFORE augmentation)…")
    before_counts, image_map = count_class_distribution(TRAIN_IMG_DIR, TRAIN_LBL_DIR)

    logger.info("Initial training instance counts:")
    for cls_id, count in before_counts.items():
        logger.info("  %s: %d", CLASS_NAMES[cls_id], count)

    target_count: int = cfg.get("target_count", 0)
    if target_count <= 0:
        target_count = max(before_counts.values()) if before_counts else 0
    logger.info("Target instance count for balancing: %d", target_count)

    if dry_run:
        for cls_id, count in before_counts.items():
            if count < target_count:
                deficit = target_count - count
                logger.info(
                    "  [DRY-RUN] %s: would generate ~%d augmented instances",
                    CLASS_NAMES[cls_id], deficit,
                )
        logger.info("DRY RUN complete — no files written.")
        return

    # 3. Augment minority classes
    output_cfg   = cfg.get("output", {})
    file_prefix  = output_cfg.get("file_prefix", "aug")
    jpeg_quality = output_cfg.get("jpeg_quality", 95)
    max_per_src  = cfg.get("max_aug_per_source", 8)

    generated_per_class: dict[int, int] = {i: 0 for i in range(NUM_CLASSES)}
    start_time = time.time()

    for cls_id, count in before_counts.items():
        if count >= target_count:
            logger.info(
                "  %s already at/above target (%d ≥ %d) — skipping",
                CLASS_NAMES[cls_id], count, target_count,
            )
            continue

        class_name = CLASS_NAMES[cls_id]
        candidates = image_map[cls_id]
        if not candidates:
            logger.warning(
                "  No candidate images for class '%s' — skipping", class_name
            )
            continue

        deficit = target_count - count
        logger.info(
            "  Augmenting minority class: '%s' (%d → %d, deficit=%d)",
            class_name, count, target_count, deficit,
        )

        aug_idx        = 0
        current_count  = count
        attempts       = 0
        max_attempts   = deficit * max_per_src * 5  # Safety cap

        while current_count < target_count and attempts < max_attempts:
            attempts += 1
            src_img_path = random.choice(candidates)
            src_lbl_path = TRAIN_LBL_DIR / f"{src_img_path.stem}.txt"

            image = cv2.imread(str(src_img_path))
            if image is None:
                continue

            yolo_boxes = load_yolo_labels(src_lbl_path)
            if not yolo_boxes:
                continue

            bboxes       = [box[1:] for box in yolo_boxes]
            class_labels = [box[0]  for box in yolo_boxes]

            try:
                augmented    = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img      = augmented["image"]
                aug_bboxes   = augmented["bboxes"]
                aug_labels   = augmented["class_labels"]
            except Exception as exc:
                logger.debug("  Augmentation failed for %s: %s", src_img_path.name, exc)
                continue

            if not aug_bboxes:
                continue  # All bboxes lost below min_visibility — retry

            # ── Construct output filename ─────────────────────────────────
            dst_stem     = f"{file_prefix}_{src_img_path.stem}_{aug_idx:05d}"
            dst_img_path = TRAIN_IMG_DIR / f"{dst_stem}.jpg"
            dst_lbl_path = TRAIN_LBL_DIR / f"{dst_stem}.txt"

            # ── Save image ────────────────────────────────────────────────
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            success = cv2.imwrite(str(dst_img_path), aug_img, encode_params)
            if not success:
                logger.warning("  Failed to write image: %s", dst_img_path)
                continue

            # ── Save labels ───────────────────────────────────────────────
            assembled: list[list] = []
            class_ids_in_aug: set[int] = set()
            for lbl, bbox in zip(aug_labels, aug_bboxes):
                assembled.append([lbl, *bbox])
                class_ids_in_aug.add(lbl)
            save_yolo_labels(dst_lbl_path, assembled)

            # ── Update counters ───────────────────────────────────────────
            if cls_id in class_ids_in_aug:
                current_count += 1
                generated_per_class[cls_id] += 1

            aug_idx += 1

        logger.info(
            "  Generated %d augmented images for '%s'. Final count: %d",
            aug_idx, class_name, current_count,
        )

    elapsed = time.time() - start_time

    # 4. Analyse AFTER distribution
    logger.info("Analysing final class distribution (AFTER augmentation)…")
    after_counts, _ = count_class_distribution(TRAIN_IMG_DIR, TRAIN_LBL_DIR)

    logger.info("Final training instance counts:")
    for cls_id, count in after_counts.items():
        logger.info("  %s: %d", CLASS_NAMES[cls_id], count)

    # 5. Save statistics
    stats_path = REPORTS_DIR / "augmentation_stats.json"
    save_augmentation_stats(
        before_counts, after_counts, generated_per_class, elapsed, stats_path
    )

    # 6. Generate charts
    if generate_charts:
        charts_dir = GRAPHS_DIR
        logger.info("Generating before/after distribution charts…")
        generate_distribution_charts(before_counts, after_counts, charts_dir)
    else:
        logger.info("Chart generation skipped (--no-charts)")

    logger.info("=" * 60)
    logger.info(
        "AUGMENTATION COMPLETE — %.1f seconds elapsed — "
        "%d new images generated",
        elapsed, sum(generated_per_class.values()),
    )
    logger.info("=" * 60)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment & balance the NEU Metal Surface Defects training split."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print class counts and deficit without generating any files.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip generating before/after distribution charts.",
    )
    args = parser.parse_args()

    cfg = load_aug_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    run_augmentation(
        cfg=cfg,
        dry_run=args.dry_run,
        generate_charts=not args.no_charts,
    )


if __name__ == "__main__":
    main()
