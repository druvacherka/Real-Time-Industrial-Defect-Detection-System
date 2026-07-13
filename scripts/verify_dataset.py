#!/usr/bin/env python3
"""
Dataset Verification Tool (Refactored)
=====================================
Real-Time Industrial Defect Detection System

Main script to verify dataset integrity and class distribution.
Utilizes the modular validation library in utils.validation.
Generates reports/dataset_validation_report.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import yaml
from pathlib import Path

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

# Initialize logging
ensure_dirs(LOGS_DIR, REPORTS_DIR)
logger = logging.getLogger("dataset_verification")
logger.setLevel(logging.INFO)

# Clear old handlers
if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "dataset_verification.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def load_class_taxonomy(classes_yaml: Path) -> dict[int, str]:
    """
    Load the accepted class mapping from classes.yaml.
    """
    if not classes_yaml.exists():
        logger.warning("classes.yaml not found at %s — falling back to config DEFECT_CLASSES", classes_yaml)
        return dict(DEFECT_CLASSES)
    try:
        with open(classes_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            names = data.get("names", [])
            if isinstance(names, list):
                return {i: name for i, name in enumerate(names)}
            elif isinstance(names, dict):
                return {int(k): v for k, v in names.items()}
    except Exception as exc:
        logger.error("Failed to parse classes.yaml: %s. Using default mapping.", exc)
    return dict(DEFECT_CLASSES)


def write_validation_report(
    report_path: Path,
    folder_status: dict[str, str],
    pair_report: dict[str, Any],
    corrupted_report: dict[str, list[str]],
    duplicates: list[dict[str, Any]],
    annotation_report: dict[str, Any],
    cross_split_duplicates: list[dict[str, Any]],
    valid_classes: dict[int, str]
) -> None:
    """
    Generate and save reports/dataset_validation_report.md
    """
    lines = [
        "# Dataset Validation Report",
        "",
        "This report provides an automated validation summary of the NEU Metal Surface Defects dataset.",
        "",
        "## 1. Directory Structure Check",
        "",
        "| Directory | Status |",
        "| --- | --- |",
    ]
    for directory, status in folder_status.items():
        icon = "✅" if status == "EXISTS" else "❌"
        lines.append(f"| `{directory}` | {icon} {status} |")
        
    lines.append("")
    lines.append("## 2. Image-Label Matching & Empty Files")
    lines.append("")
    
    missing_labels_count = sum(len(v) for v in pair_report["missing_labels"].values())
    missing_images_count = sum(len(v) for v in pair_report["missing_images"].values())
    empty_files_count = sum(len(v) for v in pair_report["empty_files"].values())
    
    lines.extend([
        f"- **Missing label files (Images without labels)**: {missing_labels_count}",
        f"- **Missing image files (Labels without images)**: {missing_images_count}",
        f"- **Empty annotation files**: {empty_files_count}",
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

    lines.append("## 3. Image Corruption Checks")
    lines.append("")
    corrupted_count = sum(len(v) for v in corrupted_report.values())
    lines.append(f"- **Corrupted/Unreadable images**: {corrupted_count}")
    if corrupted_count > 0:
        for split, files in corrupted_report.items():
            lines.append(f"  - **{split}**: {', '.join(files)}")
    lines.append("")

    lines.append("## 4. Image Duplicates Check (MD5)")
    lines.append("")
    lines.append(f"- **Duplicate image groups**: {len(duplicates)}")
    if duplicates:
        for i, dup in enumerate(duplicates[:10], start=1):
            lines.append(f"  - **Group {i}** (MD5: `{dup['md5']}`):")
            for f in dup["files"]:
                lines.append(f"    - `{f}`")
        if len(duplicates) > 10:
            lines.append(f"  - ... and {len(duplicates) - 10} more duplicate groups.")
    lines.append("")

    lines.append("## 5. Cross-Split Stem Duplicates")
    lines.append("")
    lines.append(f"- **Stems duplicated across splits**: {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        for item in cross_split_duplicates[:10]:
            lines.append(f"  - `{item['stem']}` found in {item['found_in_splits']}")
    lines.append("")

    lines.append("## 6. Annotation Consistency & Class ID Checks")
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

    if annotation_report["invalid_class_ids"]:
        lines.append("### Invalid Class IDs Found")
        for split, errs in annotation_report["invalid_class_ids"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: ID `{err['class_id']}` is invalid")
        lines.append("")

    if annotation_report["out_of_range_coords"]:
        lines.append("### Bounding Boxes Out of Range")
        for split, errs in annotation_report["out_of_range_coords"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: coordinates {err['bbox']} out of bounds")
        lines.append("")

    if annotation_report["malformed_lines"]:
        lines.append("### Malformed Annotation Lines")
        for split, errs in annotation_report["malformed_lines"].items():
            lines.append(f"- **{split}**:")
            for err in errs[:5]:
                lines.append(f"  - File `{err['file']}` line {err['line']}: `{err['raw']}`")
        lines.append("")

    lines.append("---")
    lines.append("Report generated by validation script automatically.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Successfully generated validation report at: %s", report_path)


def write_integrity_report(
    report_path: Path,
    folder_status: dict[str, str],
    pair_report: dict[str, Any],
    corrupted_report: dict[str, list[str]],
    duplicates: list[dict[str, Any]],
    annotation_report: dict[str, Any],
    cross_split_duplicates: list[dict[str, Any]],
    valid_classes: dict[int, str]
) -> None:
    """
    Generate and save reports/dataset_integrity_report.md which includes advanced box-level metrics
    (like overlapping bounding boxes and extremely small bounding boxes).
    """
    lines = [
        "# Dataset Integrity Report",
        "",
        "This report provides an advanced dataset integrity and consistency analysis of the NEU Metal Surface Defects dataset.",
        "",
        "## 1. Directory Structure",
        "",
        "| Directory | Status |",
        "| --- | --- |",
    ]
    for directory, status in folder_status.items():
        icon = "✅" if status == "EXISTS" else "❌"
        lines.append(f"| `{directory}` | {icon} {status} |")
        
    lines.append("")
    lines.append("## 2. File Mapping and Consistency")
    lines.append("")
    
    missing_labels_count = sum(len(v) for v in pair_report["missing_labels"].values())
    missing_images_count = sum(len(v) for v in pair_report["missing_images"].values())
    empty_files_count = sum(len(v) for v in pair_report["empty_files"].values())
    
    lines.extend([
        f"- **Missing labels (Images without text files)**: {missing_labels_count}",
        f"- **Missing images (Text files without image files)**: {missing_images_count}",
        f"- **Empty annotation files**: {empty_files_count}",
        ""
    ])

    lines.append("## 3. Image Integrity and Corruption Scan")
    lines.append("")
    corrupted_count = sum(len(v) for v in corrupted_report.values())
    lines.append(f"- **Corrupt or unreadable images**: {corrupted_count}")
    if corrupted_count > 0:
        for split, files in corrupted_report.items():
            lines.append(f"  - **{split}**: {', '.join(files)}")
    lines.append("")

    lines.append("## 4. Advanced Bounding Box Overlaps (IoU > 90%)")
    lines.append("")
    total_overlapping = sum(len(v) for v in annotation_report.get("overlapping_boxes", {}).values())
    lines.append(f"- **Total overlapping bounding box pairs (IoU > 0.90)**: {total_overlapping}")
    if total_overlapping > 0:
        for split, errs in annotation_report.get("overlapping_boxes", {}).items():
            lines.append(f"### {split} split:")
            for err in errs[:10]:
                lines.append(f"  - `{err['file']}`: Line {err['line1']} (class {err['class1']}) overlaps with Line {err['line2']} (class {err['class2']}) (IoU={err['iou']:.4f})")
            if len(errs) > 10:
                lines.append(f"  - ... and {len(errs) - 10} more overlapping pairs in {split}.")
    lines.append("")

    lines.append("## 5. Extremely Small Bounding Boxes (width/height < 0.005)")
    lines.append("")
    total_small = sum(len(v) for v in annotation_report.get("small_boxes", {}).values())
    lines.append(f"- **Total extremely small bounding boxes**: {total_small}")
    if total_small > 0:
        for split, errs in annotation_report.get("small_boxes", {}).items():
            lines.append(f"### {split} split:")
            for err in errs[:10]:
                lines.append(f"  - `{err['file']}`: Line {err['line']}: size [{err['bbox'][2]:.4f}, {err['bbox'][3]:.4f}]")
            if len(errs) > 10:
                lines.append(f"  - ... and {len(errs) - 10} more small boxes in {split}.")
    lines.append("")

    lines.append("## 6. Image Duplicate and Leakage Detection")
    lines.append("")
    lines.append(f"- **Duplicate image groups (MD5-based)**: {len(duplicates)}")
    lines.append(f"- **Cross-split duplicate stems (Leakage)**: {len(cross_split_duplicates)}")
    if cross_split_duplicates:
        lines.append("### Data Leakage Detected (Stems in multiple splits):")
        for item in cross_split_duplicates[:10]:
            lines.append(f"  - `{item['stem']}` found in {item['found_in_splits']}")
    lines.append("")

    lines.append("---")
    lines.append("Report generated by validation script automatically.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Successfully generated integrity report at: %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Verification and Quality Checks.")
    parser.add_argument(
        "--classes-yaml",
        type=str,
        default=str(CONFIGS_DIR / "classes.yaml"),
        help="Path to classes.yaml file."
    )
    args = parser.parse_args()

    classes_yaml_path = Path(args.classes_yaml)
    valid_classes = load_class_taxonomy(classes_yaml_path)
    valid_class_ids = set(valid_classes.keys())

    logger.info("Starting advanced dataset validation pipeline...")

    # Folder Structure Check
    folder_status = validate_folder_structure(DATASET_ROOT, SPLITS)

    # Pairs & Empty Checks
    pair_report = check_pairs_and_empty_files(IMAGES_DIR, LABELS_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Image Corruption Checks
    corrupted_report = check_corrupted_images(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Duplicate Checks
    duplicates = check_duplicate_images(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Annotation consistency Checks
    annotation_report = validate_annotation_consistency(LABELS_DIR, SPLITS, valid_class_ids)

    # Cross Split duplicates
    cross_split_duplicates = check_cross_split_duplicates(IMAGES_DIR, SPLITS, IMAGE_EXTENSIONS)

    # Output report file
    report_path = REPORTS_DIR / "dataset_validation_report.md"
    write_validation_report(
        report_path,
        folder_status,
        pair_report,
        corrupted_report,
        duplicates,
        annotation_report,
        cross_split_duplicates,
        valid_classes
    )

    integrity_report_path = REPORTS_DIR / "dataset_integrity_report.md"
    write_integrity_report(
        integrity_report_path,
        folder_status,
        pair_report,
        corrupted_report,
        duplicates,
        annotation_report,
        cross_split_duplicates,
        valid_classes
    )

    logger.info("Dataset validation pipeline completed successfully.")


if __name__ == "__main__":
    main()