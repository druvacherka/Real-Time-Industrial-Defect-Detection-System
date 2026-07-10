#!/usr/bin/env python3
"""
Automated Dataset Quality Inspection Pipeline
=============================================
Real-Time Industrial Defect Detection System

Inspects the dataset (images and labels) for corruption, duplicates, missing label files,
class ID validations, bounding box coordinate boundary consistency, and formats a markdown
report saved inside reports/quality/.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    DATASET_ROOT,
    IMAGES_DIR,
    LABELS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    CONFIGS_DIR,
    SPLITS,
    IMAGE_EXTENSIONS,
    DEFECT_CLASSES,
    ensure_dirs,
)
from utils.validation import (
    validate_folder_structure,
    check_pairs_and_empty_files,
    check_corrupted_images,
    check_duplicate_images,
    validate_annotation_consistency,
    check_cross_split_duplicates,
)

# Setup quality reporting directories
QUALITY_REPORTS_DIR = REPORTS_DIR / "quality"
ensure_dirs(LOGS_DIR, QUALITY_REPORTS_DIR)

# Initialize logger
logger = logging.getLogger("dataset_quality_inspection")
logger.setLevel(logging.INFO)

# Clear old handlers
if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "dataset_quality_inspection.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def generate_quality_report(
    report_path: Path,
    folder_status: Dict[str, str],
    pair_report: Dict[str, Any],
    corrupted_report: Dict[str, List[str]],
    duplicates: List[Dict[str, Any]],
    annotation_report: Dict[str, Any],
    cross_split_duplicates: List[Dict[str, Any]],
    valid_classes: Dict[int, str]
) -> None:
    """
    Generate and save a detailed dataset quality report in markdown format.
    """
    lines = [
        "# Dataset Quality Inspection & Integrity Report",
        "",
        "This report provides an automated validation summary of the NEU Metal Surface Defects dataset.",
        "",
        "## 1. Directory Structure Health Check",
        "",
        "| Directory | Status |",
        "| --- | --- |",
    ]
    for directory, status in folder_status.items():
        icon = "✅" if status == "EXISTS" else "❌"
        lines.append(f"| `{directory}` | {icon} {status} |")
        
    lines.append("")
    lines.append("## 2. Image-Label Pair Consistency")
    lines.append("")
    
    missing_labels_count = sum(len(v) for v in pair_report.get("missing_labels", {}).values())
    missing_images_count = sum(len(v) for v in pair_report.get("missing_images", {}).values())
    empty_files_count = sum(len(v) for v in pair_report.get("empty_files", {}).values())
    
    lines.extend([
        f"- **Missing Label Files (Images without labels)**: {missing_labels_count}",
        f"- **Missing Image Files (Labels without images)**: {missing_images_count}",
        f"- **Empty Annotation Files**: {empty_files_count}",
        ""
    ])
    
    if missing_labels_count > 0:
        lines.append("### Images Missing Labels Detail")
        for split, files in pair_report["missing_labels"].items():
            lines.append(f"- **{split}**: {', '.join(files[:10])}" + ("..." if len(files) > 10 else ""))
        lines.append("")
        
    if missing_images_count > 0:
        lines.append("### Labels Missing Images Detail")
        for split, files in pair_report["missing_images"].items():
            lines.append(f"- **{split}**: {', '.join(files[:10])}" + ("..." if len(files) > 10 else ""))
        lines.append("")

    if empty_files_count > 0:
        lines.append("### Empty Label Files Detail")
        for split, files in pair_report["empty_files"].items():
            lines.append(f"- **{split}**: {', '.join(files[:10])}" + ("..." if len(files) > 10 else ""))
        lines.append("")

    lines.append("## 3. Image Corruption Analysis")
    lines.append("")
    corrupted_count = sum(len(v) for v in corrupted_report.values())
    lines.append(f"- **Corrupted/Unreadable Images**: {corrupted_count}")
    if corrupted_count > 0:
        for split, files in corrupted_report.items():
            lines.append(f"  - **{split}**: {', '.join(files)}")
    lines.append("")

    lines.append("## 4. Image Duplication and Leakage Check")
    lines.append("")
    lines.append(f"- **Duplicate image groups (Same MD5 within or across splits)**: {len(duplicates)}")
    if duplicates:
        for i, dup in enumerate(duplicates[:10], start=1):
            lines.append(f"  - **Group {i}** (MD5: `{dup['md5']}`):")
            for f in dup["files"]:
                lines.append(f"    - `{f}`")
        if len(duplicates) > 10:
            lines.append(f"  - ... and {len(duplicates) - 10} more duplicate groups.")
    lines.append("")

    lines.append("## 5. Cross-Split Stem Duplicates (Potential Data Leakage)")
    lines.append("")
    lines.append(f"- **Stems duplicated across splits**: {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        for item in cross_split_duplicates[:10]:
            lines.append(f"  - `{item['stem']}` found in splits: {item['found_in_splits']}")
    lines.append("")

    lines.append("## 6. Bounding Box & Annotation Consistency")
    lines.append("")
    
    total_checked = annotation_report["total_annotations_checked"]
    total_errors = annotation_report["total_errors"]
    lines.extend([
        f"- **Total bounding boxes checked**: {total_checked}",
        f"- **Total consistency/class errors**: {total_errors}",
        ""
    ])

    lines.append("### Class Distribution (Valid annotations)")
    lines.append("")
    lines.append("| Class ID | Class Name | Annotations |")
    lines.append("| --- | --- | --- |")
    for cid in sorted(valid_classes.keys()):
        name = valid_classes[cid]
        count = annotation_report["class_distribution"].get(cid, 0)
        lines.append(f"| {cid} | {name} | {count} |")
    lines.append("")

    if annotation_report.get("invalid_class_ids"):
        lines.append("### Invalid Class IDs Found")
        for split, errs in annotation_report["invalid_class_ids"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: ID `{err['class_id']}` is invalid")
        lines.append("")

    if annotation_report.get("out_of_range_coords"):
        lines.append("### Bounding Boxes Out of Range (Constraint check [0.0, 1.0])")
        for split, errs in annotation_report["out_of_range_coords"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: coordinates {err['bbox']} out of bounds ({', '.join(err['issues'])})")
        lines.append("")

    if annotation_report.get("malformed_lines"):
        lines.append("### Malformed Annotation Lines")
        for split, errs in annotation_report["malformed_lines"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: `{err['raw']}`")
        lines.append("")

    lines.append("---")
    lines.append("Report generated automatically by `inspect_dataset_quality.py`.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Successfully generated dataset quality report at: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Automated Dataset Quality Inspection.")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(QUALITY_REPORTS_DIR),
        help="Directory to save the quality report."
    )
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    ensure_dirs(report_dir)

    logger.info("Starting automated dataset quality inspection pipeline...")

    # Folder Structure Check
    logger.info("Verifying folder structure...")
    folder_status = validate_folder_structure(DATASET_ROOT, SPLITS)

    # Pairs & Empty Checks
    logger.info("Checking image-label matching and empty files...")
    pair_report = check_pairs_and_empty_files(IMAGES_DIR, LABELS_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Image Corruption Checks
    logger.info("Scanning for corrupted or unreadable images...")
    corrupted_report = check_corrupted_images(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Duplicate Checks
    logger.info("Checking for duplicate images (MD5)...")
    duplicates = check_duplicate_images(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Annotation consistency Checks
    logger.info("Validating annotation format, classes, and constraints...")
    valid_class_ids = set(DEFECT_CLASSES.keys())
    annotation_report = validate_annotation_consistency(LABELS_DIR, SPLITS, valid_class_ids)

    # Cross Split duplicates
    logger.info("Checking for cross-split stem duplicates...")
    cross_split_duplicates = check_cross_split_duplicates(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Generate and save the final report
    report_path = report_dir / "dataset_quality_report.md"
    generate_quality_report(
        report_path,
        folder_status,
        pair_report,
        corrupted_report,
        duplicates,
        annotation_report,
        cross_split_duplicates,
        DEFECT_CLASSES
    )

    logger.info("Dataset quality inspection pipeline completed successfully.")


if __name__ == "__main__":
    main()
