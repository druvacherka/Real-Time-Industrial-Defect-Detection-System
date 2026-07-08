#!/usr/bin/env python3
"""
Annotation Validation Pipeline
================================
Real-Time Industrial Defect Detection System

Performs deep validation of YOLO annotation (.txt) files across all dataset
splits against the NEU Metal Surface Defect class taxonomy:

  1. Detects empty annotation files (zero-length or whitespace-only).
  2. Validates class IDs against the expected range (0 – NUM_CLASSES-1)
     and cross-checks against configs/classes.yaml.
  3. Checks that bounding-box coordinates are within the YOLO unit range
     [0.0, 1.0] and that width/height are strictly positive.
  4. Flags malformed lines (wrong token count, non-numeric values).
  5. Detects duplicate image–label stems across splits.
  6. Writes a structured JSON + Markdown report to
     reports/dataset_validation/.

Usage:
    python scripts/validate_annotations.py
    python scripts/validate_annotations.py --strict   # exit 1 on any error

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
# Bootstrap: allow running directly from the project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    PROJECT_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    CONFIGS_DIR,
    SPLITS,
    DEFECT_CLASSES,
    NUM_CLASSES,
    IMAGE_EXTENSIONS,
    ensure_dirs,
    iter_images,
    iter_labels,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
VALIDATION_REPORT_DIR: Path = REPORTS_DIR / "dataset_validation"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ensure_dirs(LOGS_DIR, VALIDATION_REPORT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "validate_annotations.log", mode="a"),
    ],
)
logger = logging.getLogger("validate_annotations")


# ===========================================================================
# Helper: load classes.yaml
# ===========================================================================

def load_classes_yaml(classes_yaml: Path) -> dict[int, str]:
    """
    Load class ID → name mapping from configs/classes.yaml.

    Expected format::

        names:
          - crazing
          - inclusion
          ...

    Falls back to DEFECT_CLASSES from config if the file is missing.

    Returns:
        Mapping of {class_id: class_name}.
    """
    if not classes_yaml.exists():
        logger.warning(
            "classes.yaml not found at %s — using built-in DEFECT_CLASSES", classes_yaml
        )
        return dict(DEFECT_CLASSES)

    try:
        with open(classes_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Failed to parse classes.yaml: %s", exc)
        return dict(DEFECT_CLASSES)

    # Support both list and dict formats
    names = data.get("names", [])
    if isinstance(names, list):
        mapping = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        mapping = {int(k): v for k, v in names.items()}
    else:
        logger.warning("Unexpected 'names' format in classes.yaml — using built-in")
        mapping = dict(DEFECT_CLASSES)

    logger.info(
        "Loaded %d classes from %s: %s",
        len(mapping),
        classes_yaml.name,
        list(mapping.values()),
    )
    return mapping


# ===========================================================================
# Annotation Validator
# ===========================================================================

class AnnotationValidator:
    """
    Deep validator for YOLO annotation files.

    Attributes
    ----------
    valid_class_ids : set[int]
        Class IDs accepted as valid (derived from classes.yaml or config).
    results : dict
        Aggregated validation results written to the final report.
    """

    def __init__(self, valid_classes: dict[int, str]) -> None:
        self.valid_class_ids: set[int] = set(valid_classes.keys())
        self.valid_classes: dict[int, str] = valid_classes

        # Per-split accumulators
        self.empty_files: dict[str, list[str]] = defaultdict(list)
        self.invalid_class_ids: dict[str, list[dict]] = defaultdict(list)
        self.out_of_range_coords: dict[str, list[dict]] = defaultdict(list)
        self.malformed_lines: dict[str, list[dict]] = defaultdict(list)
        self.duplicate_stems: list[dict] = []

        # Duplicate detection: stem → [split, ...]
        self._stem_registry: dict[str, list[str]] = defaultdict(list)

        # Overall counters
        self.total_files_checked: int = 0
        self.total_annotations_checked: int = 0
        self.total_errors: int = 0

        self.results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "valid_classes": {str(k): v for k, v in valid_classes.items()},
            "splits": {},
            "duplicate_stems": [],
            "summary": {},
        }

    # ------------------------------------------------------------------
    # Core per-file validation
    # ------------------------------------------------------------------

    def _validate_file(self, lbl_path: Path, split: str) -> dict[str, Any]:
        """
        Validate a single YOLO label file.

        Returns a dict with per-file statistics:
            - lines_checked, valid_lines, empty, errors (list of dicts)
        """
        file_result: dict[str, Any] = {
            "file": lbl_path.name,
            "split": split,
            "lines_checked": 0,
            "valid_lines": 0,
            "is_empty": False,
            "errors": [],
        }

        try:
            raw = lbl_path.read_text(encoding="utf-8")
        except Exception as exc:
            file_result["errors"].append({"type": "read_error", "detail": str(exc)})
            self.total_errors += 1
            return file_result

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        # ── Empty file check ─────────────────────────────────────────────
        if not lines:
            file_result["is_empty"] = True
            self.empty_files[split].append(lbl_path.name)
            logger.debug("  [EMPTY] %s / %s", split, lbl_path.name)
            return file_result

        for line_no, line in enumerate(lines, start=1):
            file_result["lines_checked"] += 1
            self.total_annotations_checked += 1

            parts = line.split()

            # ── Token count check ─────────────────────────────────────────
            if len(parts) != 5:
                err = {
                    "type": "malformed_line",
                    "line": line_no,
                    "expected_tokens": 5,
                    "got_tokens": len(parts),
                    "raw": line[:80],
                }
                file_result["errors"].append(err)
                self.malformed_lines[split].append(
                    {"file": lbl_path.name, **err}
                )
                self.total_errors += 1
                continue

            # ── Parse numeric values ──────────────────────────────────────
            try:
                cls_id_raw = parts[0]
                cls_id = int(float(cls_id_raw))
                cx, cy, bw, bh = (float(v) for v in parts[1:])
            except ValueError as exc:
                err = {
                    "type": "non_numeric",
                    "line": line_no,
                    "raw": line[:80],
                    "detail": str(exc),
                }
                file_result["errors"].append(err)
                self.malformed_lines[split].append(
                    {"file": lbl_path.name, **err}
                )
                self.total_errors += 1
                continue

            # ── Class ID validation ───────────────────────────────────────
            if cls_id not in self.valid_class_ids:
                err = {
                    "type": "invalid_class_id",
                    "line": line_no,
                    "class_id": cls_id,
                    "valid_range": f"0–{NUM_CLASSES - 1}",
                }
                file_result["errors"].append(err)
                self.invalid_class_ids[split].append(
                    {"file": lbl_path.name, **err}
                )
                self.total_errors += 1

            # ── Bounding-box coordinate validation ───────────────────────
            coord_errors: list[str] = []
            if not (0.0 <= cx <= 1.0):
                coord_errors.append(f"cx={cx:.4f} out of [0,1]")
            if not (0.0 <= cy <= 1.0):
                coord_errors.append(f"cy={cy:.4f} out of [0,1]")
            if not (0.0 < bw <= 1.0):
                coord_errors.append(f"bw={bw:.4f} not in (0,1]")
            if not (0.0 < bh <= 1.0):
                coord_errors.append(f"bh={bh:.4f} not in (0,1]")

            if coord_errors:
                err = {
                    "type": "bbox_out_of_range",
                    "line": line_no,
                    "issues": coord_errors,
                    "bbox": [cx, cy, bw, bh],
                }
                file_result["errors"].append(err)
                self.out_of_range_coords[split].append(
                    {"file": lbl_path.name, **err}
                )
                self.total_errors += 1

            if not file_result["errors"] or all(
                e["line"] != line_no for e in file_result["errors"]
            ):
                file_result["valid_lines"] += 1

        return file_result

    # ------------------------------------------------------------------
    # Duplicate stem detection (across all splits)
    # ------------------------------------------------------------------

    def _register_stem(self, stem: str, split: str) -> None:
        """Register an image stem for cross-split duplicate detection."""
        self._stem_registry[stem].append(split)

    def _detect_duplicates(self) -> None:
        """Flag stems appearing in more than one split."""
        for stem, splits_list in self._stem_registry.items():
            if len(splits_list) > 1:
                record = {"stem": stem, "found_in_splits": splits_list}
                self.duplicate_stems.append(record)
                logger.warning(
                    "  [DUPLICATE] stem '%s' found in splits: %s",
                    stem,
                    splits_list,
                )
        self.results["duplicate_stems"] = self.duplicate_stems
        if self.duplicate_stems:
            logger.info(
                "  Total duplicate stems detected: %d", len(self.duplicate_stems)
            )
        else:
            logger.info("  No cross-split duplicate stems found.")

    # ------------------------------------------------------------------
    # Per-split validation
    # ------------------------------------------------------------------

    def validate_split(self, split: str) -> dict[str, Any]:
        """
        Validate all annotation files in a single dataset split.

        Returns per-split aggregated statistics.
        """
        lbl_dir = LABELS_DIR / split
        img_dir = IMAGES_DIR / split

        split_result: dict[str, Any] = {
            "split": split,
            "label_dir_exists": lbl_dir.exists(),
            "image_dir_exists": img_dir.exists(),
            "total_label_files": 0,
            "total_image_files": 0,
            "empty_annotation_files": 0,
            "invalid_class_id_count": 0,
            "bbox_out_of_range_count": 0,
            "malformed_line_count": 0,
            "duplicate_stem_count": 0,
            "files": [],
        }

        if not lbl_dir.exists():
            logger.warning("  [%s] Labels directory missing: %s", split.upper(), lbl_dir)
            return split_result

        # Count images
        img_files = iter_images(img_dir) if img_dir.exists() else []
        split_result["total_image_files"] = len(img_files)

        # Register image stems for duplicate detection
        for img in img_files:
            self._register_stem(img.stem, split)

        # Validate each label file
        lbl_files = iter_labels(lbl_dir)
        split_result["total_label_files"] = len(lbl_files)
        self.total_files_checked += len(lbl_files)

        for lbl_path in lbl_files:
            file_res = self._validate_file(lbl_path, split)
            if file_res["errors"] or file_res["is_empty"]:
                split_result["files"].append(file_res)

        # Aggregate counts
        split_result["empty_annotation_files"] = len(self.empty_files.get(split, []))
        split_result["invalid_class_id_count"] = len(
            self.invalid_class_ids.get(split, [])
        )
        split_result["bbox_out_of_range_count"] = len(
            self.out_of_range_coords.get(split, [])
        )
        split_result["malformed_line_count"] = len(
            self.malformed_lines.get(split, [])
        )

        logger.info(
            "  [%s] Files: %d | Empty: %d | Bad class IDs: %d | "
            "Bbox errors: %d | Malformed lines: %d",
            split.upper(),
            len(lbl_files),
            split_result["empty_annotation_files"],
            split_result["invalid_class_id_count"],
            split_result["bbox_out_of_range_count"],
            split_result["malformed_line_count"],
        )
        return split_result

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Execute the full annotation validation pipeline across all splits.

        Returns:
            True if no critical errors were found, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("ANNOTATION VALIDATION PIPELINE")
        logger.info("=" * 60)
        logger.info(
            "Validating against %d classes: %s",
            len(self.valid_classes),
            list(self.valid_classes.values()),
        )

        for split in SPLITS:
            logger.info("-" * 40)
            logger.info("Validating split: %s", split.upper())
            logger.info("-" * 40)
            split_res = self.validate_split(split)
            self.results["splits"][split] = split_res

        # Cross-split duplicate detection
        logger.info("-" * 40)
        logger.info("Checking for cross-split duplicate stems...")
        self._detect_duplicates()

        # Build summary
        total_issues = (
            sum(len(v) for v in self.empty_files.values())
            + sum(len(v) for v in self.invalid_class_ids.values())
            + sum(len(v) for v in self.out_of_range_coords.values())
            + sum(len(v) for v in self.malformed_lines.values())
            + len(self.duplicate_stems)
        )

        self.results["summary"] = {
            "total_files_checked": self.total_files_checked,
            "total_annotations_checked": self.total_annotations_checked,
            "total_empty_files": sum(len(v) for v in self.empty_files.values()),
            "total_invalid_class_ids": sum(
                len(v) for v in self.invalid_class_ids.values()
            ),
            "total_bbox_out_of_range": sum(
                len(v) for v in self.out_of_range_coords.values()
            ),
            "total_malformed_lines": sum(
                len(v) for v in self.malformed_lines.values()
            ),
            "total_duplicate_stems": len(self.duplicate_stems),
            "total_issues": total_issues,
            "status": "PASS" if total_issues == 0 else "WARN",
        }

        logger.info("=" * 60)
        logger.info(
            "VALIDATION COMPLETE — %d issues found — Status: %s",
            total_issues,
            self.results["summary"]["status"],
        )
        logger.info("=" * 60)

        return total_issues == 0

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def save_reports(self) -> tuple[Path, Path]:
        """
        Persist validation results as JSON and Markdown.

        Returns:
            Tuple of (json_path, md_path).
        """
        ensure_dirs(VALIDATION_REPORT_DIR)

        # ── JSON ─────────────────────────────────────────────────────────
        json_path = VALIDATION_REPORT_DIR / "validation_summary.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        logger.info("JSON report saved: %s", json_path)

        # ── Markdown ──────────────────────────────────────────────────────
        md_path = VALIDATION_REPORT_DIR / "validation_report.md"
        summary = self.results["summary"]
        ts = self.results["timestamp"]
        status_icon = "✅" if summary["status"] == "PASS" else "⚠️"

        lines: list[str] = [
            "# Annotation Validation Report",
            "",
            f"> **Generated:** {ts}",
            f"> **Status:** {status_icon} {summary['status']}",
            f"> **Total Issues:** {summary['total_issues']}",
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Files checked | {summary['total_files_checked']} |",
            f"| Annotations checked | {summary['total_annotations_checked']} |",
            f"| Empty annotation files | {summary['total_empty_files']} |",
            f"| Invalid class IDs | {summary['total_invalid_class_ids']} |",
            f"| Bounding-box out of range | {summary['total_bbox_out_of_range']} |",
            f"| Malformed lines | {summary['total_malformed_lines']} |",
            f"| Cross-split duplicate stems | {summary['total_duplicate_stems']} |",
            "",
            "## Valid Class Taxonomy",
            "",
            "| ID | Class Name |",
            "|----|------------|",
        ]
        for cls_id, cls_name in self.valid_classes.items():
            lines.append(f"| {cls_id} | {cls_name} |")

        # Per-split breakdown
        for split in SPLITS:
            sr = self.results["splits"].get(split, {})
            lines += [
                "",
                f"## {split.capitalize()} Split",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Image files | {sr.get('total_image_files', 'N/A')} |",
                f"| Label files | {sr.get('total_label_files', 'N/A')} |",
                f"| Empty annotation files | {sr.get('empty_annotation_files', 0)} |",
                f"| Invalid class IDs | {sr.get('invalid_class_id_count', 0)} |",
                f"| Bbox out-of-range | {sr.get('bbox_out_of_range_count', 0)} |",
                f"| Malformed lines | {sr.get('malformed_line_count', 0)} |",
            ]

            # List problematic files (up to 10 per split)
            problem_files = sr.get("files", [])
            if problem_files:
                lines.append("")
                lines.append(f"### Files with Issues ({len(problem_files)} total)")
                lines.append("")
                for pf in problem_files[:10]:
                    err_types = ", ".join(
                        {e["type"] for e in pf.get("errors", [])}
                    ) or "empty"
                    lines.append(f"- `{pf['file']}` — {err_types}")
                if len(problem_files) > 10:
                    lines.append(f"- *… and {len(problem_files) - 10} more (see JSON)*")

        # Duplicates
        if self.duplicate_stems:
            lines += [
                "",
                "## Cross-Split Duplicate Stems",
                "",
                "| Stem | Splits |",
                "|------|--------|",
            ]
            for dup in self.duplicate_stems[:20]:
                lines.append(
                    f"| `{dup['stem']}` | {', '.join(dup['found_in_splits'])} |"
                )
            if len(self.duplicate_stems) > 20:
                lines.append(
                    f"| *… and {len(self.duplicate_stems) - 20} more* | — |"
                )

        lines += [
            "",
            "---",
            "*Report generated by `scripts/validate_annotations.py`*",
            "",
        ]

        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report saved: %s", md_path)

        return json_path, md_path


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate YOLO annotation files for the NEU defect dataset."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any validation issues are found.",
    )
    parser.add_argument(
        "--classes-yaml",
        type=str,
        default=str(CONFIGS_DIR / "classes.yaml"),
        help="Path to classes.yaml for class ID validation.",
    )
    args = parser.parse_args()

    classes_yaml_path = Path(args.classes_yaml)
    valid_classes = load_classes_yaml(classes_yaml_path)

    validator = AnnotationValidator(valid_classes)
    passed = validator.run()
    json_path, md_path = validator.save_reports()

    logger.info("JSON report : %s", json_path)
    logger.info("MD report   : %s", md_path)

    if args.strict and not passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
