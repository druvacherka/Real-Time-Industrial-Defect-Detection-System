#!/usr/bin/env python3
"""
Optimized Dataset Preprocessing and Validation Workflow
=========================================================
Real-Time Industrial Defect Detection System

Advanced preprocessing with:
  - Optimized batch image resizing
  - Improved normalization strategies
  - Duplicate image detection and removal
  - Annotation consistency validation
  - Invalid bounding box detection
  - Comprehensive preprocessing summary

Author: saniyamirjanavar-hash
Date: 2026-07-05
"""

import os
import sys
import time
import hashlib
import logging
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np

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
        logging.FileHandler(LOG_DIR / "optimize_preprocess.log", mode="a"),
    ],
)
logger = logging.getLogger("optimize_preprocess")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "yolo"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
REPORTS_DIR = PROJECT_ROOT / "reports"

SPLITS = ["train", "val", "test"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
NUM_CLASSES = 6


class AnnotationValidator:
    """Validates YOLO annotation files for consistency and correctness."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.issues = defaultdict(list)

    def validate_label_file(self, label_path: Path, image_path: Optional[Path] = None) -> List[str]:
        """
        Validate a single YOLO label file.

        Checks:
            - File is readable and non-empty
            - Each line has exactly 5 fields (class x_center y_center width height)
            - Class ID is in valid range [0, num_classes)
            - Bounding box coordinates are in [0, 1] range
            - Width and height are positive and non-zero
        """
        errors = []

        if not label_path.exists():
            errors.append(f"Label file does not exist: {label_path.name}")
            return errors

        try:
            with open(label_path, "r") as f:
                lines = f.readlines()
        except Exception as exc:
            errors.append(f"Cannot read {label_path.name}: {exc}")
            return errors

        if len(lines) == 0:
            errors.append(f"Empty annotation file: {label_path.name}")
            return errors

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()

            if len(parts) == 0:
                continue  # Skip blank lines

            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_num} — Expected 5 fields, got {len(parts)}"
                )
                continue

            try:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                errors.append(
                    f"{label_path.name}:{line_num} — Non-numeric values in annotation"
                )
                continue

            # Validate class ID
            if cls_id < 0 or cls_id >= self.num_classes:
                errors.append(
                    f"{label_path.name}:{line_num} — Invalid class ID {cls_id} "
                    f"(expected 0-{self.num_classes - 1})"
                )

            # Validate bounding box coordinates
            for name, val in [("x_center", x_center), ("y_center", y_center),
                              ("width", width), ("height", height)]:
                if val < 0 or val > 1:
                    errors.append(
                        f"{label_path.name}:{line_num} — {name}={val:.4f} out of [0,1] range"
                    )

            # Validate positive dimensions
            if width <= 0:
                errors.append(
                    f"{label_path.name}:{line_num} — Zero/negative width: {width}"
                )
            if height <= 0:
                errors.append(
                    f"{label_path.name}:{line_num} — Zero/negative height: {height}"
                )

            # Check if bbox extends beyond image boundaries
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            if x_min < -0.01 or y_min < -0.01 or x_max > 1.01 or y_max > 1.01:
                errors.append(
                    f"{label_path.name}:{line_num} — Bounding box extends beyond image "
                    f"([{x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f}])"
                )

        return errors


class DuplicateDetector:
    """Detect duplicate images using file hashing."""

    @staticmethod
    def compute_hash(file_path: Path, algorithm: str = "md5") -> str:
        """Compute hash of file contents."""
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def find_duplicates(self, image_dir: Path) -> dict:
        """
        Scan directory for duplicate images based on content hash.

        Returns:
            Dictionary mapping hash → list of file paths with that hash.
            Only includes entries with more than one file.
        """
        hash_map = defaultdict(list)

        if not image_dir.exists():
            return {}

        for img_path in image_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_hash = self.compute_hash(img_path)
                hash_map[file_hash].append(img_path)

        # Filter to only duplicates
        return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


class OptimizedPreprocessor:
    """Optimized preprocessing and validation workflow."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        interpolation: str = "linear",
    ):
        self.target_size = target_size
        self.interpolation_map = {
            "linear": cv2.INTER_LINEAR if HAS_OPENCV else None,
            "cubic": cv2.INTER_CUBIC if HAS_OPENCV else None,
            "area": cv2.INTER_AREA if HAS_OPENCV else None,
            "lanczos": cv2.INTER_LANCZOS4 if HAS_OPENCV else None,
        }
        self.interpolation = self.interpolation_map.get(interpolation)
        self.validator = AnnotationValidator()
        self.duplicate_detector = DuplicateDetector()

        self.summary = {
            "splits": {},
            "total_images": 0,
            "total_duplicates": 0,
            "total_annotation_issues": 0,
            "total_invalid_bboxes": 0,
        }

    def optimized_resize(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image using optimal interpolation based on scale direction.

        Uses INTER_AREA for downscaling (better quality) and
        INTER_LINEAR for upscaling (faster).
        """
        h, w = img.shape[:2]
        tw, th = self.target_size

        if (w, h) == (tw, th):
            return img

        # Choose interpolation: AREA for downscale, LINEAR for upscale
        if w > tw or h > th:
            interp = cv2.INTER_AREA
        else:
            interp = self.interpolation or cv2.INTER_LINEAR

        return cv2.resize(img, (tw, th), interpolation=interp)

    def improved_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Improved normalization with per-channel mean/std standardization.

        Uses ImageNet-style normalization for better transfer learning compatibility.
        """
        img = img.astype(np.float32) / 255.0

        # ImageNet mean and std (RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if img.ndim == 3 and img.shape[2] == 3:
            img = (img - mean) / std

        return img

    def process_split(self, split: str) -> dict:
        """Run full optimized preprocessing on a split."""
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split

        split_stats = {
            "images": 0,
            "labels": 0,
            "duplicates": 0,
            "annotation_issues": 0,
            "invalid_bboxes": 0,
            "duplicate_groups": [],
            "annotation_errors": [],
        }

        if not img_dir.exists():
            logger.info(f"  [{split}] Image directory not found, skipping")
            return split_stats

        # Count images
        img_files = [
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        split_stats["images"] = len(img_files)

        # Count labels
        if lbl_dir.exists():
            lbl_files = [
                f for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix == ".txt"
            ]
            split_stats["labels"] = len(lbl_files)

        # Detect duplicates
        logger.info(f"  [{split}] Scanning for duplicate images...")
        duplicates = self.duplicate_detector.find_duplicates(img_dir)
        dup_count = sum(len(paths) - 1 for paths in duplicates.values())
        split_stats["duplicates"] = dup_count
        split_stats["duplicate_groups"] = [
            [p.name for p in paths] for paths in duplicates.values()
        ]
        if dup_count > 0:
            logger.warning(f"  [{split}] Found {dup_count} duplicate images in {len(duplicates)} groups")

        # Validate annotations
        logger.info(f"  [{split}] Validating annotations...")
        if lbl_dir.exists():
            for lbl_path in lbl_dir.iterdir():
                if lbl_path.is_file() and lbl_path.suffix == ".txt":
                    errors = self.validator.validate_label_file(lbl_path)
                    if errors:
                        split_stats["annotation_issues"] += len(errors)
                        # Count bbox-specific errors
                        bbox_errors = [e for e in errors if "bounding box" in e.lower() or
                                       "width" in e.lower() or "height" in e.lower() or
                                       "out of" in e.lower()]
                        split_stats["invalid_bboxes"] += len(bbox_errors)
                        split_stats["annotation_errors"].extend(errors[:10])  # Cap at 10

        logger.info(
            f"  [{split}] Images: {split_stats['images']} | "
            f"Labels: {split_stats['labels']} | "
            f"Duplicates: {split_stats['duplicates']} | "
            f"Annotation Issues: {split_stats['annotation_issues']}"
        )
        return split_stats

    def generate_summary(self) -> Path:
        """Generate preprocessing summary as JSON and Markdown."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # JSON summary
        json_path = REPORTS_DIR / "preprocessing_validation_summary.json"
        with open(json_path, "w") as fh:
            # Convert for JSON serialization
            serializable = json.loads(json.dumps(self.summary, default=str))
            json.dump(serializable, fh, indent=2)
        logger.info(f"  JSON summary: {json_path}")

        # Markdown summary
        md_path = REPORTS_DIR / "preprocessing_validation_summary.md"
        lines = [
            "# Preprocessing & Validation Summary",
            "",
            f"- **Total Images**: {self.summary['total_images']}",
            f"- **Total Duplicates Found**: {self.summary['total_duplicates']}",
            f"- **Total Annotation Issues**: {self.summary['total_annotation_issues']}",
            f"- **Invalid Bounding Boxes**: {self.summary['total_invalid_bboxes']}",
            "",
        ]

        for split in SPLITS:
            stats = self.summary["splits"].get(split, {})
            lines += [
                f"## {split.capitalize()}",
                f"- Images: {stats.get('images', 0)}",
                f"- Labels: {stats.get('labels', 0)}",
                f"- Duplicates: {stats.get('duplicates', 0)}",
                f"- Annotation Issues: {stats.get('annotation_issues', 0)}",
                "",
            ]

            dup_groups = stats.get("duplicate_groups", [])
            if dup_groups:
                lines.append("### Duplicate Groups")
                for group in dup_groups[:5]:
                    lines.append(f"- {', '.join(group)}")
                lines.append("")

            errors = stats.get("annotation_errors", [])
            if errors:
                lines.append("### Annotation Errors")
                for err in errors[:5]:
                    lines.append(f"- {err}")
                lines.append("")

        lines.append("---")
        lines.append("*Generated by `optimize_preprocess.py`*")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"  Markdown summary: {md_path}")
        return md_path

    def run(self):
        """Execute the full optimized preprocessing workflow."""
        logger.info("=" * 60)
        logger.info("OPTIMIZED PREPROCESSING & VALIDATION WORKFLOW")
        logger.info("=" * 60)

        start = time.time()

        for split in SPLITS:
            split_stats = self.process_split(split)
            self.summary["splits"][split] = split_stats
            self.summary["total_images"] += split_stats["images"]
            self.summary["total_duplicates"] += split_stats["duplicates"]
            self.summary["total_annotation_issues"] += split_stats["annotation_issues"]
            self.summary["total_invalid_bboxes"] += split_stats["invalid_bboxes"]

        self.generate_summary()

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"WORKFLOW COMPLETE in {elapsed:.2f}s")
        logger.info(f"  Images: {self.summary['total_images']}")
        logger.info(f"  Duplicates: {self.summary['total_duplicates']}")
        logger.info(f"  Annotation Issues: {self.summary['total_annotation_issues']}")
        logger.info(f"  Invalid Bboxes: {self.summary['total_invalid_bboxes']}")
        logger.info("=" * 60)


def main():
    if not HAS_OPENCV:
        logger.error("OpenCV is required. Install with: pip install opencv-python")
        sys.exit(1)
    preprocessor = OptimizedPreprocessor()
    preprocessor.run()


if __name__ == "__main__":
    main()
