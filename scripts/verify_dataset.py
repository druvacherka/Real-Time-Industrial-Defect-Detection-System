#!/usr/bin/env python3
"""
Dataset Verification and Quality Analysis Tool
================================================
Real-Time Industrial Defect Detection System

Performs comprehensive verification of the NEU Metal Surface Defects dataset:
  - Folder structure validation
  - Missing image/label detection
  - Corrupted image identification
  - Per-class image counting
  - Dataset quality report generation

Author: saniyamirjanavar-hash
Date: 2026-07-04
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

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
        logging.FileHandler(LOG_DIR / "dataset_verification.log", mode="a"),
    ],
)
logger = logging.getLogger("dataset_verification")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
YOLO_ROOT = DATASET_ROOT / "yolo"
IMAGES_DIR = YOLO_ROOT / "images"
LABELS_DIR = YOLO_ROOT / "labels"
RAW_DIR = DATASET_ROOT / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]

# NEU-DET defect classes
DEFECT_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


class DatasetVerifier:
    """Comprehensive dataset verification and quality analysis."""

    def __init__(self, dataset_root: Path = DATASET_ROOT):
        self.dataset_root = dataset_root
        self.yolo_root = dataset_root / "yolo"
        self.images_dir = self.yolo_root / "images"
        self.labels_dir = self.yolo_root / "labels"
        self.raw_dir = dataset_root / "raw"
        self.reports_dir = PROJECT_ROOT / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Verification results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "folder_checks": {},
            "missing_labels": {},
            "missing_images": {},
            "corrupted_images": {},
            "class_distribution": {},
            "split_statistics": {},
            "total_images": 0,
            "total_labels": 0,
            "total_corrupted": 0,
            "overall_status": "PASS",
        }

    def verify_folder_structure(self) -> bool:
        """Verify that all required dataset directories exist."""
        logger.info("=" * 60)
        logger.info("STEP 1: Verifying folder structure")
        logger.info("=" * 60)

        required_dirs = [
            self.dataset_root,
            self.yolo_root,
            self.images_dir,
            self.labels_dir,
        ]

        for split in SPLITS:
            required_dirs.append(self.images_dir / split)
            required_dirs.append(self.labels_dir / split)

        all_exist = True
        for dir_path in required_dirs:
            relative = dir_path.relative_to(PROJECT_ROOT)
            if dir_path.exists():
                logger.info(f"  [OK] {relative}")
                self.results["folder_checks"][str(relative)] = "EXISTS"
            else:
                logger.warning(f"  [MISSING] {relative}")
                self.results["folder_checks"][str(relative)] = "MISSING"
                all_exist = False

        if not all_exist:
            self.results["overall_status"] = "WARN"
        return all_exist

    def detect_missing_pairs(self) -> dict:
        """Detect images without labels and labels without images."""
        logger.info("=" * 60)
        logger.info("STEP 2: Detecting missing image-label pairs")
        logger.info("=" * 60)

        summary = {"missing_labels": 0, "missing_images": 0}

        for split in SPLITS:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            if not img_dir.exists() or not lbl_dir.exists():
                logger.warning(f"  [{split.upper()}] Skipped — directory missing")
                continue

            img_stems = {
                f.stem: f
                for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            }
            lbl_stems = {
                f.stem: f
                for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".txt"
            }

            # Images missing labels
            no_label = sorted(img_stems.keys() - lbl_stems.keys())
            if no_label:
                logger.warning(
                    f"  [{split.upper()}] {len(no_label)} images have no label file"
                )
                self.results["missing_labels"][split] = no_label
                summary["missing_labels"] += len(no_label)

            # Labels missing images
            no_image = sorted(lbl_stems.keys() - img_stems.keys())
            if no_image:
                logger.warning(
                    f"  [{split.upper()}] {len(no_image)} labels have no image file"
                )
                self.results["missing_images"][split] = no_image
                summary["missing_images"] += len(no_image)

            if not no_label and not no_image:
                logger.info(
                    f"  [{split.upper()}] All {len(img_stems)} pairs matched"
                )

        if summary["missing_labels"] > 0 or summary["missing_images"] > 0:
            self.results["overall_status"] = "WARN"
        return summary

    def identify_corrupted_images(self) -> int:
        """Attempt to read every image and flag those that fail."""
        logger.info("=" * 60)
        logger.info("STEP 3: Identifying corrupted images")
        logger.info("=" * 60)

        if not HAS_OPENCV:
            logger.warning("  OpenCV not installed — skipping corruption check")
            return 0

        corrupted_count = 0
        for split in SPLITS:
            img_dir = self.images_dir / split
            if not img_dir.exists():
                continue

            split_corrupted = []
            image_files = [
                f
                for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]

            for img_path in image_files:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        raise ValueError("cv2.imread returned None")
                    if img.shape[0] == 0 or img.shape[1] == 0:
                        raise ValueError("Zero-dimension image")
                except Exception as exc:
                    logger.warning(f"  [CORRUPTED] {img_path.name}: {exc}")
                    split_corrupted.append(img_path.name)
                    corrupted_count += 1

            if split_corrupted:
                self.results["corrupted_images"][split] = split_corrupted
            else:
                logger.info(
                    f"  [{split.upper()}] No corrupted images ({len(image_files)} checked)"
                )

        self.results["total_corrupted"] = corrupted_count
        if corrupted_count > 0:
            self.results["overall_status"] = "FAIL"
        return corrupted_count

    def count_per_class(self) -> dict:
        """Parse label files and count bounding boxes per class."""
        logger.info("=" * 60)
        logger.info("STEP 4: Counting images per class")
        logger.info("=" * 60)

        class_counts = defaultdict(int)
        images_per_class = defaultdict(set)

        for split in SPLITS:
            lbl_dir = self.labels_dir / split
            if not lbl_dir.exists():
                continue

            label_files = [
                f for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".txt"
            ]

            for lbl_path in label_files:
                try:
                    with open(lbl_path, "r") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                class_counts[cls_id] += 1
                                images_per_class[cls_id].add(lbl_path.stem)
                except Exception as exc:
                    logger.warning(f"  Could not parse {lbl_path.name}: {exc}")

        # Also count raw images per class folder (NEU-DET structure)
        raw_neu = self.raw_dir / "NEU-DET"
        if raw_neu.exists():
            for subset in ["train", "validation"]:
                subset_dir = raw_neu / subset / "images"
                if not subset_dir.exists():
                    continue
                for class_dir in subset_dir.iterdir():
                    if class_dir.is_dir():
                        count = len([
                            f for f in class_dir.iterdir()
                            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                        ])
                        if count > 0:
                            images_per_class[class_dir.name] = set()
                            class_counts[class_dir.name] = count
                            logger.info(
                                f"  [RAW] {class_dir.name}: {count} images"
                            )

        # Convert sets to counts for JSON serialization
        distribution = {}
        for cls_id, count in sorted(class_counts.items(), key=lambda x: str(x[0])):
            cls_name = DEFECT_CLASSES.get(cls_id, str(cls_id))
            distribution[cls_name] = count
            logger.info(f"  Class '{cls_name}': {count} annotations")

        self.results["class_distribution"] = distribution
        return distribution

    def compute_split_statistics(self) -> dict:
        """Compute per-split statistics (image count, label count)."""
        logger.info("=" * 60)
        logger.info("STEP 5: Computing split statistics")
        logger.info("=" * 60)

        stats = {}
        total_imgs = 0
        total_lbls = 0

        for split in SPLITS:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            img_count = 0
            lbl_count = 0

            if img_dir.exists():
                img_count = len([
                    f for f in img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ])
            if lbl_dir.exists():
                lbl_count = len([
                    f for f in lbl_dir.iterdir()
                    if f.is_file() and f.suffix.lower() == ".txt"
                ])

            stats[split] = {"images": img_count, "labels": lbl_count}
            total_imgs += img_count
            total_lbls += lbl_count
            logger.info(
                f"  [{split.upper()}] Images: {img_count} | Labels: {lbl_count}"
            )

        self.results["split_statistics"] = stats
        self.results["total_images"] = total_imgs
        self.results["total_labels"] = total_lbls
        return stats

    def generate_report(self) -> Path:
        """Generate a Markdown quality report and save it."""
        logger.info("=" * 60)
        logger.info("STEP 6: Generating dataset quality report")
        logger.info("=" * 60)

        report_path = self.reports_dir / "dataset_quality_report.md"
        status_emoji = {
            "PASS": "✅",
            "WARN": "⚠️",
            "FAIL": "❌",
        }
        status = self.results["overall_status"]

        lines = [
            "# Dataset Quality Report",
            "",
            f"> Generated: {self.results['timestamp']}",
            f"> Overall Status: {status_emoji.get(status, '❓')} **{status}**",
            "",
            "---",
            "",
            "## Folder Structure",
            "",
            "| Directory | Status |",
            "|-----------|--------|",
        ]

        for folder, st in self.results["folder_checks"].items():
            icon = "✅" if st == "EXISTS" else "❌"
            lines.append(f"| `{folder}` | {icon} {st} |")

        lines += ["", "## Split Statistics", ""]
        lines += ["| Split | Images | Labels |", "|-------|--------|--------|"]
        for split, st in self.results.get("split_statistics", {}).items():
            lines.append(f"| {split} | {st['images']} | {st['labels']} |")
        lines.append(
            f"| **Total** | **{self.results['total_images']}** "
            f"| **{self.results['total_labels']}** |"
        )

        # Class distribution
        if self.results["class_distribution"]:
            lines += ["", "## Class Distribution", ""]
            lines += ["| Class | Count |", "|-------|-------|"]
            for cls, cnt in self.results["class_distribution"].items():
                lines.append(f"| {cls} | {cnt} |")

        # Missing pairs
        total_missing = sum(
            len(v) for v in self.results["missing_labels"].values()
        ) + sum(len(v) for v in self.results["missing_images"].values())
        if total_missing > 0:
            lines += ["", "## Missing Pairs", ""]
            for split, items in self.results["missing_labels"].items():
                lines.append(f"- **{split}**: {len(items)} images missing labels")
            for split, items in self.results["missing_images"].items():
                lines.append(f"- **{split}**: {len(items)} labels missing images")

        # Corrupted images
        if self.results["total_corrupted"] > 0:
            lines += [
                "",
                "## Corrupted Images",
                "",
                f"Total corrupted: **{self.results['total_corrupted']}**",
                "",
            ]
            for split, items in self.results["corrupted_images"].items():
                lines.append(f"### {split}")
                for name in items[:10]:
                    lines.append(f"- `{name}`")

        lines += ["", "---", f"*Report generated by `verify_dataset.py`*", ""]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"  Report saved to {report_path}")

        # Also save raw JSON results
        json_path = self.reports_dir / "dataset_quality_report.json"
        with open(json_path, "w") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        logger.info(f"  JSON results saved to {json_path}")

        return report_path

    def run(self) -> bool:
        """Execute the full verification pipeline."""
        logger.info("Starting dataset verification pipeline...")
        logger.info(f"Dataset root: {self.dataset_root}")

        self.verify_folder_structure()
        self.detect_missing_pairs()
        self.identify_corrupted_images()
        self.count_per_class()
        self.compute_split_statistics()
        report = self.generate_report()

        logger.info("=" * 60)
        logger.info(f"VERIFICATION COMPLETE — Status: {self.results['overall_status']}")
        logger.info(f"Report: {report}")
        logger.info("=" * 60)

        return self.results["overall_status"] != "FAIL"


def main():
    """Entry point for dataset verification."""
    verifier = DatasetVerifier()
    success = verifier.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()