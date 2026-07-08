#!/usr/bin/env python3
"""
Dataset Verification and Quality Analysis Tool  (v2)
=====================================================
Real-Time Industrial Defect Detection System

Performs comprehensive verification of the NEU Metal Surface Defects dataset:

  1. Folder structure validation
  2. Missing image/label pair detection
  3. Empty annotation file detection
  4. Corrupted image identification (via OpenCV)
  5. Duplicate image detection (MD5 hash-based, within and across splits)
  6. Class ID validation against configs/classes.yaml
  7. Per-class annotation counting & split statistics
  8. Dataset quality report generation (Markdown + JSON)

Reports saved to:
    reports/dataset_validation/

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --no-hash    # skip MD5 duplicate check

Author: saniyamirjanavar-hash
Date:   2026-07-08  feat(data): improve dataset validation and annotation
                    verification pipeline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: allow running directly from any working directory
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    PROJECT_ROOT,
    DATASET_ROOT,
    YOLO_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    RAW_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    CONFIGS_DIR,
    DEFECT_CLASSES,
    SPLITS,
    IMAGE_EXTENSIONS as SUPPORTED_IMAGE_EXTENSIONS,
    ensure_dirs,
    iter_images,
    iter_labels,
    class_name as get_class_name,
)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ---------------------------------------------------------------------------
# Output directory for validation reports
# ---------------------------------------------------------------------------
VALIDATION_REPORT_DIR: Path = REPORTS_DIR / "dataset_validation"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
ensure_dirs(LOGS_DIR, VALIDATION_REPORT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "dataset_verification.log", mode="a"),
    ],
)
logger = logging.getLogger("dataset_verification")


# ===========================================================================
# Helper: load class taxonomy from classes.yaml
# ===========================================================================

def load_valid_class_ids(classes_yaml: Path) -> dict[int, str]:
    """
    Load the accepted class ID → name mapping from configs/classes.yaml.

    Supports both list and dict formats:
        names: [crazing, inclusion, ...]
        names: {0: crazing, 1: inclusion, ...}

    Falls back to DEFECT_CLASSES from config if the file is missing or
    cannot be parsed.

    Args:
        classes_yaml: Absolute path to classes.yaml.

    Returns:
        Mapping of {class_id: class_name}.
    """
    if not classes_yaml.exists():
        logger.warning(
            "classes.yaml not found at %s — falling back to built-in DEFECT_CLASSES",
            classes_yaml,
        )
        return dict(DEFECT_CLASSES)

    try:
        with open(classes_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to parse classes.yaml: %s — using built-in fallback", exc)
        return dict(DEFECT_CLASSES)

    names = data.get("names", [])
    if isinstance(names, list):
        mapping = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        mapping = {int(k): v for k, v in names.items()}
    else:
        logger.warning("Unexpected 'names' format — using built-in DEFECT_CLASSES")
        mapping = dict(DEFECT_CLASSES)

    logger.info(
        "Loaded %d valid classes from %s: %s",
        len(mapping), classes_yaml.name, list(mapping.values()),
    )
    return mapping


# ===========================================================================
# Dataset Verifier
# ===========================================================================

class DatasetVerifier:
    """
    Comprehensive dataset verification and quality analysis.

    Attributes
    ----------
    dataset_root : Path
        Root directory of the dataset (contains yolo/, raw/, …).
    valid_class_ids : set[int]
        Class IDs accepted as valid per classes.yaml.
    run_hash_check : bool
        Whether to perform MD5-based duplicate image detection.
    results : dict
        Aggregated results that are serialised to the final reports.
    """

    def __init__(
        self,
        dataset_root: Path = DATASET_ROOT,
        valid_classes: dict[int, str] | None = None,
        run_hash_check: bool = True,
    ) -> None:
        self.dataset_root = dataset_root
        self.yolo_root = dataset_root / "yolo"
        self.images_dir = self.yolo_root / "images"
        self.labels_dir = self.yolo_root / "labels"
        self.raw_dir = dataset_root / "raw"
        self.valid_classes: dict[int, str] = (
            valid_classes if valid_classes is not None else dict(DEFECT_CLASSES)
        )
        self.valid_class_ids: set[int] = set(self.valid_classes.keys())
        self.run_hash_check = run_hash_check

        # Aggregated result store
        self.results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "folder_checks": {},
            "missing_labels": {},
            "missing_images": {},
            "empty_annotation_files": {},
            "corrupted_images": {},
            "duplicate_images": [],
            "invalid_class_ids": {},
            "class_distribution": {},
            "split_statistics": {},
            "total_images": 0,
            "total_labels": 0,
            "total_corrupted": 0,
            "overall_status": "PASS",
        }

    # ------------------------------------------------------------------
    # Step 1: Folder structure
    # ------------------------------------------------------------------

    def verify_folder_structure(self) -> bool:
        """Verify that all required dataset directories exist."""
        logger.info("=" * 60)
        logger.info("STEP 1: Verifying folder structure")
        logger.info("=" * 60)

        required_dirs: list[Path] = [
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
            try:
                relative = dir_path.relative_to(PROJECT_ROOT)
            except ValueError:
                relative = dir_path

            if dir_path.exists():
                logger.info("  [OK]      %s", relative)
                self.results["folder_checks"][str(relative)] = "EXISTS"
            else:
                logger.warning("  [MISSING] %s", relative)
                self.results["folder_checks"][str(relative)] = "MISSING"
                all_exist = False

        if not all_exist:
            self.results["overall_status"] = "WARN"
        return all_exist

    # ------------------------------------------------------------------
    # Step 2: Missing image–label pairs + empty annotation detection
    # ------------------------------------------------------------------

    def detect_missing_pairs(self) -> dict[str, int]:
        """
        Detect images without labels, labels without images,
        and empty annotation files.

        Returns:
            Summary counts {missing_labels, missing_images, empty_files}.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Detecting missing image-label pairs & empty files")
        logger.info("=" * 60)

        summary: dict[str, int] = {
            "missing_labels": 0,
            "missing_images": 0,
            "empty_files": 0,
        }

        for split in SPLITS:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            if not img_dir.exists() or not lbl_dir.exists():
                logger.warning("  [%s] Skipped — directory missing", split.upper())
                continue

            img_stems: dict[str, Path] = {
                f.stem: f
                for f in img_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            }
            lbl_stems: dict[str, Path] = {
                f.stem: f
                for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".txt"
            }

            # Images missing labels
            no_label = sorted(img_stems.keys() - lbl_stems.keys())
            if no_label:
                logger.warning(
                    "  [%s] %d image(s) have no label file", split.upper(), len(no_label)
                )
                self.results["missing_labels"][split] = no_label
                summary["missing_labels"] += len(no_label)

            # Labels missing images
            no_image = sorted(lbl_stems.keys() - img_stems.keys())
            if no_image:
                logger.warning(
                    "  [%s] %d label(s) have no image file", split.upper(), len(no_image)
                )
                self.results["missing_images"][split] = no_image
                summary["missing_images"] += len(no_image)

            # Empty annotation files
            empty: list[str] = []
            for stem, lbl_path in lbl_stems.items():
                try:
                    content = lbl_path.read_text(encoding="utf-8").strip()
                    if not content:
                        empty.append(lbl_path.name)
                except Exception:
                    empty.append(lbl_path.name)
            if empty:
                logger.warning(
                    "  [%s] %d empty annotation file(s) detected", split.upper(), len(empty)
                )
                self.results["empty_annotation_files"][split] = empty
                summary["empty_files"] += len(empty)

            if not no_label and not no_image and not empty:
                logger.info(
                    "  [%s] All %d image-label pairs matched, 0 empty files",
                    split.upper(), len(img_stems),
                )

        total_issues = sum(summary.values())
        if total_issues > 0:
            self.results["overall_status"] = "WARN"
        return summary

    # ------------------------------------------------------------------
    # Step 3: Corrupted image detection
    # ------------------------------------------------------------------

    def identify_corrupted_images(self) -> int:
        """
        Attempt to decode every image file via OpenCV; flag failures.

        Returns:
            Total count of corrupted images found.
        """
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

            split_corrupted: list[str] = []
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
                        raise ValueError("Zero-dimension image detected")
                except Exception as exc:
                    logger.warning(
                        "  [CORRUPTED] %s/%s — %s", split, img_path.name, exc
                    )
                    split_corrupted.append(img_path.name)
                    corrupted_count += 1

            if split_corrupted:
                self.results["corrupted_images"][split] = split_corrupted
            else:
                logger.info(
                    "  [%s] No corrupted images (%d checked)",
                    split.upper(), len(image_files),
                )

        self.results["total_corrupted"] = corrupted_count
        if corrupted_count > 0:
            self.results["overall_status"] = "FAIL"
        return corrupted_count

    # ------------------------------------------------------------------
    # Step 4: Duplicate image detection (MD5 hash)
    # ------------------------------------------------------------------

    def detect_duplicate_images(self) -> list[dict]:
        """
        Detect duplicate images across all splits using MD5 checksums.

        Two files are considered duplicates when their MD5 digest is
        identical, regardless of filename.

        Returns:
            List of duplicate groups
            [{"md5": "…", "files": ["split/name.jpg", …]}, …].
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Detecting duplicate images (MD5)")
        logger.info("=" * 60)

        hash_map: dict[str, list[str]] = defaultdict(list)

        for split in SPLITS:
            img_dir = self.images_dir / split
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    try:
                        digest = hashlib.md5(img_path.read_bytes()).hexdigest()
                        hash_map[digest].append(f"{split}/{img_path.name}")
                    except Exception as exc:
                        logger.warning(
                            "  Could not hash %s/%s: %s", split, img_path.name, exc
                        )

        duplicates: list[dict] = [
            {"md5": md5, "files": sorted(paths)}
            for md5, paths in hash_map.items()
            if len(paths) > 1
        ]

        self.results["duplicate_images"] = duplicates
        if duplicates:
            total_dup_files = sum(len(d["files"]) for d in duplicates)
            logger.warning(
                "  %d duplicate group(s) found — %d total duplicate files",
                len(duplicates), total_dup_files,
            )
            for dup in duplicates[:5]:
                logger.warning("    MD5 %s: %s", dup["md5"][:12], dup["files"])
            if len(duplicates) > 5:
                logger.warning("    … and %d more groups", len(duplicates) - 5)
            self.results["overall_status"] = "WARN"
        else:
            logger.info("  No duplicate images found.")

        return duplicates

    # ------------------------------------------------------------------
    # Step 5: Class ID validation
    # ------------------------------------------------------------------

    def validate_class_ids(self) -> dict[str, list[dict]]:
        """
        Scan all label files and flag annotations with class IDs outside
        the valid set defined in classes.yaml.

        Returns:
            Per-split list of invalid class ID occurrences.
        """
        logger.info("=" * 60)
        logger.info("STEP 5: Validating class IDs against classes.yaml")
        logger.info("=" * 60)
        logger.info(
            "  Valid class IDs: %s",
            {k: v for k, v in self.valid_classes.items()},
        )

        invalid_per_split: dict[str, list[dict]] = defaultdict(list)

        for split in SPLITS:
            lbl_dir = self.labels_dir / split
            if not lbl_dir.exists():
                continue

            lbl_files = [
                f for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".txt"
            ]

            for lbl_path in lbl_files:
                try:
                    with open(lbl_path, "r", encoding="utf-8") as fh:
                        for line_no, line in enumerate(fh, start=1):
                            parts = line.strip().split()
                            if not parts:
                                continue
                            try:
                                cls_id = int(float(parts[0]))
                            except ValueError:
                                continue
                            if cls_id not in self.valid_class_ids:
                                record = {
                                    "file": lbl_path.name,
                                    "line": line_no,
                                    "class_id": cls_id,
                                    "expected": sorted(self.valid_class_ids),
                                }
                                invalid_per_split[split].append(record)
                                logger.warning(
                                    "  [INVALID CLASS] %s/%s line %d: "
                                    "class_id=%d not in %s",
                                    split, lbl_path.name, line_no,
                                    cls_id, sorted(self.valid_class_ids),
                                )
                except Exception as exc:
                    logger.warning(
                        "  Could not read %s/%s: %s", split, lbl_path.name, exc
                    )

            n_invalid = len(invalid_per_split.get(split, []))
            if n_invalid == 0:
                logger.info(
                    "  [%s] All class IDs valid (%d label files checked)",
                    split.upper(), len(lbl_files),
                )
            else:
                logger.warning(
                    "  [%s] %d invalid class ID annotation(s) found",
                    split.upper(), n_invalid,
                )

        self.results["invalid_class_ids"] = {
            split: records for split, records in invalid_per_split.items()
        }

        total_invalid = sum(len(v) for v in invalid_per_split.values())
        if total_invalid > 0:
            self.results["overall_status"] = "WARN"

        return dict(invalid_per_split)

    # ------------------------------------------------------------------
    # Step 6: Per-class annotation counting
    # ------------------------------------------------------------------

    def count_per_class(self) -> dict[str, int]:
        """
        Parse all label files across all splits and count bounding boxes
        per class.

        Returns:
            Mapping of {class_name: annotation_count}.
        """
        logger.info("=" * 60)
        logger.info("STEP 6: Counting annotations per class")
        logger.info("=" * 60)

        class_counts: dict[int, int] = defaultdict(int)

        for split in SPLITS:
            lbl_dir = self.labels_dir / split
            if not lbl_dir.exists():
                continue

            for lbl_path in lbl_dir.iterdir():
                if not (lbl_path.is_file() and lbl_path.suffix.lower() == ".txt"):
                    continue
                try:
                    with open(lbl_path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    cls_id = int(float(parts[0]))
                                    class_counts[cls_id] += 1
                                except ValueError:
                                    pass
                except Exception as exc:
                    logger.warning(
                        "  Could not parse %s/%s: %s", split, lbl_path.name, exc
                    )

        distribution: dict[str, int] = {}
        for cls_id in sorted(class_counts.keys()):
            name = self.valid_classes.get(cls_id, f"unknown_{cls_id}")
            count = class_counts[cls_id]
            distribution[name] = count
            logger.info("  Class '%s' (id=%d): %d annotations", name, cls_id, count)

        self.results["class_distribution"] = distribution
        return distribution

    # ------------------------------------------------------------------
    # Step 7: Split statistics
    # ------------------------------------------------------------------

    def compute_split_statistics(self) -> dict[str, dict]:
        """
        Compute per-split image and label counts.

        Returns:
            {split: {"images": N, "labels": N}}
        """
        logger.info("=" * 60)
        logger.info("STEP 7: Computing split statistics")
        logger.info("=" * 60)

        stats: dict[str, dict] = {}
        total_imgs = 0
        total_lbls = 0

        for split in SPLITS:
            img_dir = self.images_dir / split
            lbl_dir = self.labels_dir / split

            img_count = 0
            lbl_count = 0

            if img_dir.exists():
                img_count = sum(
                    1 for f in img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                )
            if lbl_dir.exists():
                lbl_count = sum(
                    1 for f in lbl_dir.iterdir()
                    if f.is_file() and f.suffix.lower() == ".txt"
                )

            stats[split] = {"images": img_count, "labels": lbl_count}
            total_imgs += img_count
            total_lbls += lbl_count
            logger.info(
                "  [%s] Images: %d | Labels: %d", split.upper(), img_count, lbl_count
            )

        self.results["split_statistics"] = stats
        self.results["total_images"] = total_imgs
        self.results["total_labels"] = total_lbls
        return stats

    # ------------------------------------------------------------------
    # Step 8: Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> tuple[Path, Path]:
        """
        Write a Markdown quality report and a JSON summary to
        reports/dataset_validation/.

        Returns:
            Tuple of (markdown_path, json_path).
        """
        logger.info("=" * 60)
        logger.info("STEP 8: Generating dataset quality report")
        logger.info("=" * 60)

        ensure_dirs(VALIDATION_REPORT_DIR)

        # ── JSON ─────────────────────────────────────────────────────────
        json_path = VALIDATION_REPORT_DIR / "dataset_quality_report.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        logger.info("  JSON report saved: %s", json_path)

        # ── Markdown ──────────────────────────────────────────────────────
        md_path = VALIDATION_REPORT_DIR / "dataset_quality_report.md"
        status = self.results["overall_status"]
        status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(status, "❓")

        lines: list[str] = [
            "# Dataset Quality Report",
            "",
            f"> **Generated:** {self.results['timestamp']}",
            f"> **Overall Status:** {status_icon} **{status}**",
            "",
            "---",
            "",
            "## 1. Folder Structure",
            "",
            "| Directory | Status |",
            "|-----------|--------|",
        ]
        for folder, st in self.results["folder_checks"].items():
            icon = "✅" if st == "EXISTS" else "❌"
            lines.append(f"| `{folder}` | {icon} {st} |")

        # Split statistics
        lines += [
            "",
            "## 2. Split Statistics",
            "",
            "| Split | Images | Labels |",
            "|-------|--------|--------|",
        ]
        for split, st in self.results.get("split_statistics", {}).items():
            lines.append(f"| {split} | {st['images']} | {st['labels']} |")
        lines.append(
            f"| **Total** | **{self.results['total_images']}** "
            f"| **{self.results['total_labels']}** |"
        )

        # Class distribution
        if self.results["class_distribution"]:
            lines += [
                "",
                "## 3. Class Distribution",
                "",
                "| Class | Annotations |",
                "|-------|-------------|",
            ]
            for cls, cnt in self.results["class_distribution"].items():
                lines.append(f"| {cls} | {cnt} |")

        # Missing pairs
        total_missing_lbl = sum(
            len(v) for v in self.results["missing_labels"].values()
        )
        total_missing_img = sum(
            len(v) for v in self.results["missing_images"].values()
        )
        total_empty = sum(
            len(v) for v in self.results.get("empty_annotation_files", {}).values()
        )

        lines += [
            "",
            "## 4. Data Integrity",
            "",
            "| Check | Result |",
            "|-------|--------|",
            f"| Images missing labels | {'✅ 0' if total_missing_lbl == 0 else f'⚠️ {total_missing_lbl}'} |",
            f"| Labels missing images | {'✅ 0' if total_missing_img == 0 else f'⚠️ {total_missing_img}'} |",
            f"| Empty annotation files | {'✅ 0' if total_empty == 0 else f'⚠️ {total_empty}'} |",
            f"| Corrupted images | {'✅ 0' if self.results['total_corrupted'] == 0 else f'❌ {self.results[\"total_corrupted\"]}'} |",
            f"| Duplicate images | {'✅ 0 groups' if not self.results['duplicate_images'] else f'⚠️ {len(self.results[\"duplicate_images\"])} group(s)'} |",
        ]

        # Invalid class IDs
        total_invalid = sum(
            len(v) for v in self.results.get("invalid_class_ids", {}).values()
        )
        lines.append(
            f"| Invalid class IDs | {'✅ 0' if total_invalid == 0 else f'⚠️ {total_invalid}'} |"
        )

        # Duplicate details
        if self.results["duplicate_images"]:
            lines += [
                "",
                "## 5. Duplicate Images",
                "",
                "| MD5 (first 12) | Files |",
                "|----------------|-------|",
            ]
            for dup in self.results["duplicate_images"][:15]:
                lines.append(
                    f"| `{dup['md5'][:12]}` | {', '.join(dup['files'])} |"
                )

        # Invalid class ID details
        if total_invalid > 0:
            lines += ["", "## 6. Invalid Class ID Details", ""]
            for split, records in self.results.get("invalid_class_ids", {}).items():
                if records:
                    lines.append(f"### {split.capitalize()} ({len(records)} occurrences)")
                    for r in records[:10]:
                        lines.append(
                            f"- `{r['file']}` line {r['line']}: "
                            f"class_id={r['class_id']}"
                        )
                    if len(records) > 10:
                        lines.append(f"  *… and {len(records) - 10} more (see JSON)*")

        lines += [
            "",
            "---",
            "*Report generated by `scripts/verify_dataset.py`*",
            "",
        ]

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("  Markdown report saved: %s", md_path)
        return md_path, json_path

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Execute the full verification pipeline (steps 1–8).

        Returns:
            True if overall status is not FAIL.
        """
        logger.info("Starting dataset verification pipeline…")
        logger.info("Dataset root: %s", self.dataset_root)
        logger.info(
            "Valid classes (%d): %s",
            len(self.valid_classes),
            list(self.valid_classes.values()),
        )

        self.verify_folder_structure()
        self.detect_missing_pairs()
        self.identify_corrupted_images()

        if self.run_hash_check:
            self.detect_duplicate_images()
        else:
            logger.info("  Skipping MD5 duplicate check (--no-hash)")

        self.validate_class_ids()
        self.count_per_class()
        self.compute_split_statistics()
        md_path, json_path = self.generate_report()

        logger.info("=" * 60)
        logger.info(
            "VERIFICATION COMPLETE — Status: %s", self.results["overall_status"]
        )
        logger.info("  Markdown : %s", md_path)
        logger.info("  JSON     : %s", json_path)
        logger.info("=" * 60)

        return self.results["overall_status"] != "FAIL"


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify NEU Metal Surface Defects dataset integrity."
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip the MD5 duplicate-image detection step.",
    )
    parser.add_argument(
        "--classes-yaml",
        type=str,
        default=str(CONFIGS_DIR / "classes.yaml"),
        help="Path to classes.yaml used for class ID validation.",
    )
    args = parser.parse_args()

    valid_classes = load_valid_class_ids(Path(args.classes_yaml))

    verifier = DatasetVerifier(
        valid_classes=valid_classes,
        run_hash_check=not args.no_hash,
    )
    success = verifier.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()