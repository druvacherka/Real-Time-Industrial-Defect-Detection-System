#!/usr/bin/env python3
"""
Shared Configuration Module
============================
Real-Time Industrial Defect Detection System

Central source of truth for all paths, constants, and class definitions
used across the dataset pipeline scripts.  Import this module instead of
duplicating constants in every script.

Usage:
    from scripts.config import PROJECT_ROOT, DEFECT_CLASSES, SPLITS
    # or when running a script directly from the project root:
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from config import PROJECT_ROOT, DEFECT_CLASSES, SPLITS

Author: saniyamirjanavar-hash
Date: 2026-07-07
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Layout
# ---------------------------------------------------------------------------
# This file lives at  <project_root>/scripts/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_ROOT: Path = PROJECT_ROOT / "dataset"
YOLO_ROOT: Path    = DATASET_ROOT / "yolo"
IMAGES_DIR: Path   = YOLO_ROOT / "images"
LABELS_DIR: Path   = YOLO_ROOT / "labels"
RAW_DIR: Path      = DATASET_ROOT / "raw"
AUGMENTED_DIR: Path = DATASET_ROOT / "augmented"
PROCESSED_DIR: Path = DATASET_ROOT / "processed"

REPORTS_DIR: Path  = PROJECT_ROOT / "reports"
GRAPHS_DIR: Path   = REPORTS_DIR / "graphs"
LOGS_DIR: Path     = PROJECT_ROOT / "logs"

CONFIGS_DIR: Path  = PROJECT_ROOT / "configs"
DATA_YAML: Path    = CONFIGS_DIR / "data.yaml"
AUG_YAML: Path     = CONFIGS_DIR / "augmentation.yaml"

DOCS_DIR: Path     = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Dataset Splits
# ---------------------------------------------------------------------------
SPLITS: list[str] = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Defect Classes — NEU-DET (6 classes)
# ---------------------------------------------------------------------------
DEFECT_CLASSES: dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

CLASS_NAMES: list[str] = [DEFECT_CLASSES[i] for i in range(len(DEFECT_CLASSES))]
NUM_CLASSES: int = len(DEFECT_CLASSES)

# ---------------------------------------------------------------------------
# File Extensions
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
LABEL_EXTENSION: str = ".txt"

# ---------------------------------------------------------------------------
# Preprocessing Defaults
# ---------------------------------------------------------------------------
TARGET_SIZE: tuple[int, int] = (640, 640)   # (width, height)

# ---------------------------------------------------------------------------
# Visualization Color Palette — one hex color per class
# ---------------------------------------------------------------------------
CLASS_COLORS: list[str] = [
    "#FF6B6B",   # crazing        — coral red
    "#4ECDC4",   # inclusion      — teal
    "#45B7D1",   # patches        — sky blue
    "#96CEB4",   # pitted_surface — sage green
    "#FFEAA7",   # rolled-in_scale — yellow
    "#DDA0DD",   # scratches      — plum
]

# OpenCV BGR equivalents for CLASS_COLORS (used when drawing on images)
CLASS_COLORS_BGR: list[tuple[int, int, int]] = [
    (107, 107, 255),   # crazing
    (196, 205, 78),    # inclusion
    (209, 183, 69),    # patches
    (180, 206, 150),   # pitted_surface
    (167, 234, 255),   # rolled-in_scale
    (221, 160, 221),   # scratches
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    """Create directories (and parents) if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def iter_images(directory: Path) -> list[Path]:
    """Return a sorted list of image files in *directory*."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def iter_labels(directory: Path) -> list[Path]:
    """Return a sorted list of label (.txt) files in *directory*."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix == LABEL_EXTENSION
    )


def class_name(cls_id: int) -> str:
    """Return the human-readable name for a class ID."""
    return DEFECT_CLASSES.get(cls_id, f"unknown_{cls_id}")
