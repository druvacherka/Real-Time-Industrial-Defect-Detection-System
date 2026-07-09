#!/usr/bin/env python3
"""
Dataset Balancing and Augmentation Tool (Refactored)
===================================================
Real-Time Industrial Defect Detection System

Main script to balance dataset classes in YOLO training split.
Utilizes the modular augmentation library in utils.augmentation.
Generates reports/dataset_balancing_report.md.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    IMAGES_DIR,
    LABELS_DIR,
    REPORTS_DIR,
    GRAPHS_DIR,
    LOGS_DIR,
    AUG_YAML,
    CLASS_NAMES,
    ensure_dirs,
)
from utils.augmentation import (
    build_augmentation_pipeline,
    balance_dataset_classes,
    write_balancing_report,
)

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

# Initialize logging
ensure_dirs(LOGS_DIR, REPORTS_DIR, GRAPHS_DIR)
logger = logging.getLogger("augment_dataset")
logger.setLevel(logging.INFO)

if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "augment_dataset.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)

TRAIN_IMG_DIR = IMAGES_DIR / "train"
TRAIN_LBL_DIR = LABELS_DIR / "train"


def load_config() -> dict:
    """Load augmentation config."""
    if not HAS_YAML or not AUG_YAML.exists():
        logger.warning("Could not load configs/augmentation.yaml, using defaults")
        return {}
    try:
        with open(AUG_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.error("Failed to parse config: %s", exc)
        return {}


def generate_charts(before: dict[int, int], after: dict[int, int]) -> None:
    """Generate side-by-side distribution charts."""
    if not HAS_MPL:
        return
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    before_counts = [before.get(i, 0) for i in range(len(CLASS_NAMES))]
    after_counts = [after.get(i, 0) for i in range(len(CLASS_NAMES))]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, before_counts, width, label="Before Augmentation", color="#3498db")
    ax.bar(x + width/2, after_counts, width, label="After Augmentation", color="#2ecc71")
    
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Class Balancing Distribution - Before vs After")
    ax.legend()
    
    chart_path = GRAPHS_DIR / "augmentation_before_after_bar.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved balancing bar chart at %s", chart_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Augmentation and Balancing.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print balancing plan without executing."
    )
    args = parser.parse_args()

    cfg = load_config()
    
    # Random seed
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    if args.dry_run:
        logger.info("DRY-RUN mode enabled.")
        
    transform = build_augmentation_pipeline(cfg)
    
    # Target count from config (0 means auto-detect from majority class)
    target_count = cfg.get("target_count", 0)

    stats = balance_dataset_classes(
        TRAIN_IMG_DIR,
        TRAIN_LBL_DIR,
        target_count,
        cfg,
        transform,
        CLASS_NAMES
    )

    # Save balancing report
    report_path = REPORTS_DIR / "dataset_balancing_report.md"
    write_balancing_report(report_path, stats, CLASS_NAMES)

    # Save stats json
    stats_json_path = REPORTS_DIR / "augmentation_stats.json"
    import json
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Generate charts
    generate_charts(stats["before"], stats["after"])

    logger.info("Dataset Balancing pipeline completed successfully.")


if __name__ == "__main__":
    main()
