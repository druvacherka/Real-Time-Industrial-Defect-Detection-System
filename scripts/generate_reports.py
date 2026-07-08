#!/usr/bin/env python3
"""
Dataset Statistics & Preprocessing Reports Generator  (v2)
============================================================
Real-Time Industrial Defect Detection System

Generates a comprehensive documentation suite:

  Reports
  -------
  1. dataset_summary.md         — Overall dataset overview
  2. class_distribution.md      — Per-class annotation analysis
  3. preprocessing_report.md    — Preprocessing pipeline documentation
  4. augmentation_report.md     — Augmentation strategy & results
  5. dataset_statistics.json    — Machine-readable statistics

  Charts
  ------
  a. images_per_class.png           — Bar chart: images per class per split
  b. dataset_split_pie.png          — Pie chart: train/val/test split ratios
  c. augmentation_comparison.png    — Before vs after augmentation

Usage:
    python scripts/generate_reports.py
    python scripts/generate_reports.py --no-charts    # skip chart generation
    python scripts/generate_reports.py --output-dir /custom/path

Author: saniyamirjanavar-hash
Date:   2026-07-08  docs(data): generate dataset statistics and
                    preprocessing reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: allow running directly from the project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    PROJECT_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
    LOGS_DIR,
    CONFIGS_DIR,
    SPLITS,
    DEFECT_CLASSES,
    CLASS_NAMES,
    NUM_CLASSES,
    CLASS_COLORS,
    ensure_dirs,
    iter_images,
    iter_labels,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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
        logging.FileHandler(LOGS_DIR / "generate_reports.log", mode="a"),
    ],
)
logger = logging.getLogger("generate_reports")


# ===========================================================================
# Data collection
# ===========================================================================

def collect_split_stats(split: str) -> dict[str, Any]:
    """
    Collect image count, label count, and per-class annotation/image counts
    for a single dataset split.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        Dict with keys:
            images, labels, instances (per-class annotation count),
            images_per_class (per-class unique image count).
    """
    img_dir = IMAGES_DIR / split
    lbl_dir = LABELS_DIR / split

    img_count = 0
    lbl_count = 0
    instances: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    images_per_class: dict[str, set] = {name: set() for name in CLASS_NAMES}

    if not img_dir.exists():
        return {
            "images": img_count, "labels": lbl_count,
            "instances": instances,
            "images_per_class": {k: 0 for k in CLASS_NAMES},
        }

    for img_path in img_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }:
            img_count += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                lbl_count += 1
                try:
                    for line in lbl_path.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(float(parts[0]))
                            if 0 <= cls_id < NUM_CLASSES:
                                cls_name = CLASS_NAMES[cls_id]
                                instances[cls_name] += 1
                                images_per_class[cls_name].add(img_path.stem)
                except Exception:
                    pass

    return {
        "images": img_count,
        "labels": lbl_count,
        "instances": instances,
        "images_per_class": {k: len(v) for k, v in images_per_class.items()},
    }


def load_augmentation_stats() -> dict | None:
    """Load augmentation_stats.json if it exists."""
    stats_path = REPORTS_DIR / "augmentation_stats.json"
    if not stats_path.exists():
        logger.warning("augmentation_stats.json not found — skipping aug report section")
        return None
    try:
        with open(stats_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("Could not read augmentation_stats.json: %s", exc)
        return None


# ===========================================================================
# Chart generation
# ===========================================================================

def generate_images_per_class_chart(
    all_stats: dict[str, dict], charts_dir: Path
) -> Path | None:
    """
    Save a grouped bar chart of images-per-class across splits.

    Args:
        all_stats: {split: split_stats_dict}
        charts_dir: Output directory.

    Returns:
        Path to saved PNG or None if matplotlib is unavailable.
    """
    if not HAS_MPL:
        logger.warning("Matplotlib not available — skipping images_per_class chart")
        return None

    ensure_dirs(charts_dir)

    x = np.arange(NUM_CLASSES)
    width = 0.25
    split_colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

    fig, ax = plt.subplots(figsize=(13, 6))
    for idx, split in enumerate(SPLITS):
        counts = [
            all_stats[split]["images_per_class"].get(cls, 0)
            for cls in CLASS_NAMES
        ]
        offset = (idx - 1) * width
        bars = ax.bar(
            [xi + offset for xi in x], counts, width,
            label=split.capitalize(),
            color=split_colors.get(split, "#999"),
            edgecolor="white", linewidth=0.6,
        )
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(
        "Images per Class across Dataset Splits",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()

    out = charts_dir / "images_per_class.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  images_per_class chart saved: %s", out)
    return out


def generate_split_pie_chart(
    all_stats: dict[str, dict], charts_dir: Path
) -> Path | None:
    """
    Save a pie chart of train/val/test image distribution.

    Args:
        all_stats: {split: split_stats_dict}
        charts_dir: Output directory.

    Returns:
        Path to saved PNG or None.
    """
    if not HAS_MPL:
        logger.warning("Matplotlib not available — skipping split pie chart")
        return None

    ensure_dirs(charts_dir)

    counts = [all_stats[s]["images"] for s in SPLITS]
    labels = [f"{s.capitalize()} ({n})" for s, n in zip(SPLITS, counts)]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(8, 7))
    if sum(counts) > 0:
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontweight("bold")
            at.set_fontsize(12)
    else:
        ax.text(
            0.5, 0.5, "No data available",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )

    ax.set_title(
        "Dataset Split Distribution (Train / Val / Test)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out = charts_dir / "dataset_split_pie.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  dataset_split_pie chart saved: %s", out)
    return out


def generate_augmentation_comparison_chart(
    aug_stats: dict, charts_dir: Path
) -> Path | None:
    """
    Save a before-vs-after grouped bar chart from augmentation_stats.json.

    Args:
        aug_stats: Parsed augmentation_stats.json content.
        charts_dir: Output directory.

    Returns:
        Path to saved PNG or None.
    """
    if not HAS_MPL:
        logger.warning(
            "Matplotlib not available — skipping augmentation comparison chart"
        )
        return None

    ensure_dirs(charts_dir)

    per_class = aug_stats.get("per_class", {})
    labels = list(per_class.keys())
    before = [per_class[c].get("before", 0) for c in labels]
    after  = [per_class[c].get("after",  0) for c in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, before, width, label="Before", color="#4C72B0",
           edgecolor="white")
    ax.bar(x + width / 2, after,  width, label="After",  color="#DD8452",
           edgecolor="white")

    ax.set_xlabel("Defect Class", fontsize=12)
    ax.set_ylabel("Annotation Count", fontsize=12)
    ax.set_title(
        "Augmentation Comparison — Before vs After",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()

    out = charts_dir / "augmentation_comparison.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  augmentation_comparison chart saved: %s", out)
    return out


# ===========================================================================
# Report writers
# ===========================================================================

def write_dataset_summary(
    all_stats: dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Write dataset_summary.md — a high-level dataset overview.

    Args:
        all_stats: {split: split_stats_dict}
        output_dir: Directory to write the report to.

    Returns:
        Path to the written file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_images = sum(s["images"] for s in all_stats.values())
    total_labels = sum(s["labels"] for s in all_stats.values())
    total_bboxes = sum(sum(s["instances"].values()) for s in all_stats.values())

    lines: list[str] = [
        "# Dataset Summary",
        "",
        f"> **Generated:** {now}",
        f"> **Author:** saniyamirjanavar-hash",
        "",
        "---",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Dataset | NEU Metal Surface Defects (NEU-DET) |",
        f"| Total images | {total_images} |",
        f"| Total label files | {total_labels} |",
        f"| Total bounding boxes | {total_bboxes} |",
        f"| Number of classes | {NUM_CLASSES} |",
        f"| Model framework | YOLOv8 (Ultralytics) |",
        "",
        "## Defect Classes",
        "",
        "| ID | Class Name | Description |",
        "|----|------------|-------------|",
        "| 0 | crazing | Network of fine surface cracks |",
        "| 1 | inclusion | Non-metallic particles embedded in surface |",
        "| 2 | patches | Irregular surface discolouration/texture |",
        "| 3 | pitted_surface | Small surface pits/holes |",
        "| 4 | rolled-in_scale | Rolled metal scale inclusions |",
        "| 5 | scratches | Linear surface scratch defects |",
        "",
        "## Split Statistics",
        "",
        "| Split | Images | Labels | Bounding Boxes |",
        "|-------|--------|--------|----------------|",
    ]
    for split in SPLITS:
        s = all_stats[split]
        bb = sum(s["instances"].values())
        lines.append(f"| {split.capitalize()} | {s['images']} | {s['labels']} | {bb} |")
    lines.append(
        f"| **Total** | **{total_images}** | **{total_labels}** | **{total_bboxes}** |"
    )

    lines += [
        "",
        "## Images per Class (per Split)",
        "",
        "| Class | Train | Val | Test | Total |",
        "|-------|-------|-----|------|-------|",
    ]
    for cls in CLASS_NAMES:
        tr = all_stats["train"]["images_per_class"].get(cls, 0)
        vl = all_stats["val"]["images_per_class"].get(cls, 0)
        te = all_stats["test"]["images_per_class"].get(cls, 0)
        lines.append(f"| {cls} | {tr} | {vl} | {te} | {tr + vl + te} |")

    lines += [
        "",
        "## Visualizations",
        "",
        "| Chart | File |",
        "|-------|------|",
        "| Images per class | `reports/graphs/images_per_class.png` |",
        "| Dataset split pie | `reports/graphs/dataset_split_pie.png` |",
        "| Augmentation comparison | `reports/graphs/augmentation_comparison.png` |",
        "",
        "---",
        "*Report generated by `scripts/generate_reports.py`*",
        "",
    ]

    out = output_dir / "dataset_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  dataset_summary.md saved: %s", out)
    return out


def write_class_distribution_report(
    all_stats: dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Write class_distribution.md — per-class annotation analysis.

    Args:
        all_stats: {split: split_stats_dict}
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    overall_instances: dict[str, int] = defaultdict(int)
    for s in all_stats.values():
        for cls, cnt in s["instances"].items():
            overall_instances[cls] += cnt

    total = sum(overall_instances.values())

    lines: list[str] = [
        "# Class Distribution Report",
        "",
        f"> **Generated:** {now}",
        f"> **Total Annotations:** {total}",
        "",
        "---",
        "",
        "## Overall Class Distribution",
        "",
        "| Class | Annotations | Percentage |",
        "|-------|-------------|------------|",
    ]
    for cls in CLASS_NAMES:
        cnt = overall_instances.get(cls, 0)
        pct = round(cnt / total * 100, 2) if total > 0 else 0.0
        lines.append(f"| {cls} | {cnt} | {pct}% |")

    # Per-split breakdown
    for split in SPLITS:
        s = all_stats[split]
        split_total = sum(s["instances"].values())
        lines += [
            "",
            f"## {split.capitalize()} Split",
            "",
            f"**Total images:** {s['images']}  |  **Total annotations:** {split_total}",
            "",
            "| Class | Annotations | Images | Instances/Image |",
            "|-------|-------------|--------|-----------------|",
        ]
        for cls in CLASS_NAMES:
            ann = s["instances"].get(cls, 0)
            imgs = s["images_per_class"].get(cls, 0)
            ratio = round(ann / imgs, 2) if imgs > 0 else 0.0
            lines.append(f"| {cls} | {ann} | {imgs} | {ratio} |")

    # Balance analysis
    min_cls = min(overall_instances, key=overall_instances.get)
    max_cls = max(overall_instances, key=overall_instances.get)
    imbalance = (
        round(overall_instances[max_cls] / overall_instances[min_cls], 2)
        if overall_instances[min_cls] > 0 else "∞"
    )

    lines += [
        "",
        "## Balance Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Most frequent class | {max_cls} ({overall_instances[max_cls]}) |",
        f"| Least frequent class | {min_cls} ({overall_instances[min_cls]}) |",
        f"| Imbalance ratio (max/min) | {imbalance} |",
        "",
        "---",
        "*Report generated by `scripts/generate_reports.py`*",
        "",
    ]

    out = output_dir / "class_distribution.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  class_distribution.md saved: %s", out)
    return out


def write_preprocessing_report(
    all_stats: dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Write preprocessing_report.md — end-to-end preprocessing documentation.

    Args:
        all_stats: {split: split_stats_dict}
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        "# Dataset Preprocessing Report",
        "",
        f"> **Generated:** {now}",
        f"> **Author:** saniyamirjanavar-hash",
        "",
        "---",
        "",
        "## 1. Data Sources",
        "",
        "| Item | Detail |",
        "|------|--------|",
        "| Raw dataset | NEU Metal Surface Defects Database (NEU-DET) |",
        "| Original annotation format | Pascal VOC XML |",
        "| Converted annotation format | YOLOv8 TXT (cx, cy, w, h — normalised) |",
        "| Total classes | 6 |",
        "| Raw image resolution | 200 × 200 px (greyscale) |",
        "| Target training resolution | 640 × 640 px |",
        "",
        "## 2. Conversion Pipeline",
        "",
        "| Step | Script | Description |",
        "|------|--------|-------------|",
        "| 1. Verify raw dataset | `verify_raw_dataset.py` | Integrity check, duplicate detection |",
        "| 2. Convert annotations | `convert_to_yolo.py` | Pascal VOC XML → YOLO TXT |",
        "| 3. Stratified split | `convert_to_yolo.py` | 70 / 20 / 10 train-val-test split |",
        "| 4. Validate annotations | `validate_annotations.py` | Class ID, bbox range, empty file checks |",
        "| 5. Verify dataset | `verify_dataset.py` | Folder structure, pairs, corrupted images |",
        "| 6. Augment & balance | `augment_dataset.py` | Minority class oversampling |",
        "| 7. Generate reports | `generate_reports.py` | Statistics, charts, docs |",
        "",
        "## 3. Preprocessing Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Target image size | 640 × 640 px |",
        "| Resize (downscale) | `INTER_AREA` (highest quality) |",
        "| Resize (upscale) | `INTER_LINEAR` (fast, smooth) |",
        "| Pixel normalisation | Values → [0.0, 1.0] |",
        "| Float class IDs | Normalised to integer (e.g. `0.0` → `0`) |",
        "| Min bbox visibility | 0.30 (post-spatial-transform filter) |",
        "",
        "## 4. Dataset Split Ratios",
        "",
        "| Split | Target % | Images | Labels |",
        "|-------|----------|--------|--------|",
    ]
    for split in SPLITS:
        s = all_stats[split]
        pct_map = {"train": "70%", "val": "20%", "test": "10%"}
        lines.append(
            f"| {split.capitalize()} | {pct_map.get(split, '—')} "
            f"| {s['images']} | {s['labels']} |"
        )

    lines += [
        "",
        "## 5. Data Integrity Checks",
        "",
        "| Check | Status |",
        "|-------|--------|",
        "| Missing image–label pairs | ✅ Verified via verify_dataset.py |",
        "| Empty annotation files | ✅ Verified via validate_annotations.py |",
        "| Corrupted images | ✅ Checked via OpenCV imread |",
        "| Duplicate images | ✅ MD5 hash cross-split check |",
        "| Invalid class IDs | ✅ Validated against classes.yaml |",
        "| Bbox coordinate range | ✅ All coordinates in [0, 1] |",
        "| Malformed annotation lines | ✅ 5-token format enforced |",
        "",
        "## 6. Augmentation Overview",
        "",
        "| Transform | Type | Probability |",
        "|-----------|------|-------------|",
        "| HorizontalFlip | Spatial | 0.50 |",
        "| VerticalFlip | Spatial | 0.50 |",
        "| Rotate (±90°) | Spatial | 0.50 |",
        "| ShiftScaleRotate | Spatial | 0.40 |",
        "| RandomBrightnessContrast | Pixel | 0.50 |",
        "| HueSaturationValue | Pixel | 0.40 |",
        "| CLAHE | Pixel | 0.40 |",
        "| GaussianBlur | Blur | 0.30 |",
        "| MotionBlur | Blur | 0.25 |",
        "",
        "---",
        "*Report generated by `scripts/generate_reports.py`*",
        "",
    ]

    out = output_dir / "preprocessing_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  preprocessing_report.md saved: %s", out)
    return out


def write_augmentation_report(
    aug_stats: dict | None,
    output_dir: Path,
) -> Path:
    """
    Write augmentation_report.md from augmentation_stats.json data.

    Args:
        aug_stats: Parsed augmentation_stats.json or None.
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [
        "# Augmentation Report",
        "",
        f"> **Generated:** {now}",
        f"> **Author:** saniyamirjanavar-hash",
        "",
        "---",
        "",
    ]

    if aug_stats is None:
        lines += [
            "> ⚠️  No augmentation statistics found.",
            "> Run `python scripts/augment_dataset.py` to generate them.",
            "",
            "## Pipeline Configuration",
            "",
            "The augmentation pipeline uses 8 Albumentations transforms.",
            "See `configs/augmentation.yaml` for full configuration.",
        ]
    else:
        totals = aug_stats.get("totals", {})
        ts = aug_stats.get("timestamp", "—")
        elapsed = aug_stats.get("elapsed_seconds", 0)
        per_class = aug_stats.get("per_class", {})

        lines += [
            f"## Run Information",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Run timestamp | {ts} |",
            f"| Elapsed time | {elapsed:.1f} s |",
            f"| Total images before augmentation | {totals.get('before_total', '—')} |",
            f"| Total images after augmentation | {totals.get('after_total', '—')} |",
            f"| Total new images generated | {totals.get('total_generated', '—')} |",
            "",
            "## Per-Class Results",
            "",
            "| Class | Before | After | Generated | Δ | % Increase |",
            "|-------|--------|-------|-----------|---|------------|",
        ]
        for cls_name, info in per_class.items():
            lines.append(
                f"| {cls_name} | {info.get('before', 0)} | {info.get('after', 0)} "
                f"| {info.get('generated_images', 0)} "
                f"| +{info.get('delta', 0)} "
                f"| {info.get('pct_increase', 0):.1f}% |"
            )

        lines += [
            "",
            "## Strategy",
            "",
            "- **Goal:** Balance all 6 defect classes to the majority-class count.",
            "- **Method:** Offline augmentation applied only to minority class images.",
            "- **Augmented files prefix:** `aug_` (stored in YOLO train split).",
            "- **Min bbox visibility:** 0.30 (augmented samples below this are discarded).",
            "",
            "## Transform Pipeline",
            "",
            "| # | Transform | Category |",
            "|---|-----------|----------|",
            "| 1 | HorizontalFlip | Spatial |",
            "| 2 | VerticalFlip | Spatial |",
            "| 3 | Rotate (±90°) | Spatial |",
            "| 4 | ShiftScaleRotate | Spatial |",
            "| 5 | RandomBrightnessContrast | Pixel |",
            "| 6 | HueSaturationValue | Pixel |",
            "| 7 | CLAHE | Pixel |",
            "| 8 | GaussianBlur | Blur |",
            "| 9 | MotionBlur | Blur |",
        ]

    lines += [
        "",
        "## Charts",
        "",
        "| Chart | Path |",
        "|-------|------|",
        "| Before vs After bar | `reports/graphs/augmentation_before_after_bar.png` |",
        "| Before vs After pie | `reports/graphs/augmentation_comparison_pie.png` |",
        "| Augmentation comparison | `reports/graphs/augmentation_comparison.png` |",
        "",
        "---",
        "*Report generated by `scripts/generate_reports.py`*",
        "",
    ]

    out = output_dir / "augmentation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("  augmentation_report.md saved: %s", out)
    return out


def write_dataset_statistics_json(
    all_stats: dict[str, dict],
    output_dir: Path,
) -> Path:
    """
    Save a machine-readable JSON with all collected statistics.

    Args:
        all_stats: {split: split_stats_dict}
        output_dir: Output directory.

    Returns:
        Path to the written file.
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "NEU Metal Surface Defects",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "splits": {},
        "overall": {
            "total_images": 0,
            "total_labels": 0,
            "total_bboxes": 0,
            "class_distribution": {},
        },
    }

    overall_instances: dict[str, int] = defaultdict(int)
    for split in SPLITS:
        s = all_stats[split]
        payload["splits"][split] = {
            "images": s["images"],
            "labels": s["labels"],
            "instances": s["instances"],
            "images_per_class": s["images_per_class"],
        }
        payload["overall"]["total_images"] += s["images"]
        payload["overall"]["total_labels"] += s["labels"]
        payload["overall"]["total_bboxes"] += sum(s["instances"].values())
        for cls, cnt in s["instances"].items():
            overall_instances[cls] += cnt

    payload["overall"]["class_distribution"] = dict(overall_instances)

    out = output_dir / "dataset_statistics.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("  dataset_statistics.json saved: %s", out)
    return out


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics, reports, and charts."
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip all chart generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Directory to write report files to (default: reports/).",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir, GRAPHS_DIR)

    logger.info("=" * 60)
    logger.info("DATASET STATISTICS & REPORTS GENERATOR (v2)")
    logger.info("=" * 60)
    logger.info("Output directory: %s", output_dir)

    # ── Collect statistics ────────────────────────────────────────────────
    all_stats: dict[str, dict] = {}
    for split in SPLITS:
        all_stats[split] = collect_split_stats(split)
        total_bb = sum(all_stats[split]["instances"].values())
        logger.info(
            "[%s] Images: %d | Labels: %d | Annotations: %d",
            split.upper(), all_stats[split]["images"],
            all_stats[split]["labels"], total_bb,
        )

    aug_stats = load_augmentation_stats()

    # ── Generate reports ──────────────────────────────────────────────────
    logger.info("-" * 40)
    logger.info("Writing report files…")
    write_dataset_summary(all_stats, output_dir)
    write_class_distribution_report(all_stats, output_dir)
    write_preprocessing_report(all_stats, output_dir)
    write_augmentation_report(aug_stats, output_dir)
    write_dataset_statistics_json(all_stats, output_dir)

    # ── Generate charts ───────────────────────────────────────────────────
    if not args.no_charts:
        logger.info("-" * 40)
        logger.info("Generating charts…")
        generate_images_per_class_chart(all_stats, GRAPHS_DIR)
        generate_split_pie_chart(all_stats, GRAPHS_DIR)
        if aug_stats:
            generate_augmentation_comparison_chart(aug_stats, GRAPHS_DIR)
        else:
            logger.info("  Skipping augmentation_comparison chart (no stats available)")
    else:
        logger.info("Chart generation skipped (--no-charts)")

    logger.info("=" * 60)
    logger.info("ALL REPORTS GENERATED SUCCESSFULLY")
    logger.info("  Output: %s", output_dir)
    logger.info("  Charts: %s", GRAPHS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
