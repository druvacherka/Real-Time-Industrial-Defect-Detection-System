#!/usr/bin/env python3
"""
Master Dataset Build & Orchestration Script
=============================================
Real-Time Industrial Defect Detection System

Runs the complete dataset finalization pipeline in the correct order:
  1. Verify folder structure and raw dataset integrity
  2. Normalize float class IDs in all label files  (0.0 → 0)
  3. Run augmentation / class-balancing on the training split
  4. Validate all image-label pairs (post-augmentation)
  5. Generate class distribution statistics + charts
  6. Write the final data.yaml with correct absolute path
  7. Print a comprehensive dataset summary

Usage (from project root):
    python scripts/build_final_dataset.py            # full pipeline
    python scripts/build_final_dataset.py --dry-run  # stats only, no augmentation

Author: saniyamirjanavar-hash
Date:   2026-07-07
"""

from __future__ import annotations

import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Bootstrap: allow running from project root without installing the package
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    PROJECT_ROOT, DATASET_ROOT, YOLO_ROOT, IMAGES_DIR, LABELS_DIR,
    REPORTS_DIR, GRAPHS_DIR, LOGS_DIR, DATA_YAML, AUG_YAML,
    SPLITS, DEFECT_CLASSES, NUM_CLASSES, IMAGE_EXTENSIONS,
    CLASS_NAMES, ensure_dirs,
)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ensure_dirs(LOGS_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "build_final_dataset.log", mode="a"),
    ],
)
logger = logging.getLogger("build_final_dataset")

BANNER = "=" * 65


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def step_verify_structure() -> bool:
    """Verify that all required YOLO split directories exist."""
    logger.info(f"{BANNER}")
    logger.info("STEP 1 — Verify folder structure")
    logger.info(BANNER)

    required = [YOLO_ROOT, IMAGES_DIR, LABELS_DIR]
    for split in SPLITS:
        required += [IMAGES_DIR / split, LABELS_DIR / split]

    all_ok = True
    for d in required:
        if d.exists():
            logger.info(f"  [OK]      {d.relative_to(PROJECT_ROOT)}")
        else:
            logger.error(f"  [MISSING] {d.relative_to(PROJECT_ROOT)}")
            all_ok = False

    return all_ok


def step_fix_float_class_ids() -> int:
    """Normalize class IDs from float (0.0) to integer (0) in all label files."""
    logger.info(f"{BANNER}")
    logger.info("STEP 2 — Fix float class IDs in label files")
    logger.info(BANNER)

    total_fixed_lines = 0
    total_files_touched = 0

    for split in SPLITS:
        lbl_dir = LABELS_DIR / split
        if not lbl_dir.exists():
            continue
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            try:
                lines = lbl_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            fixed: list[str] = []
            file_modified = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls_id = int(float(parts[0]))
                        new_line = f"{cls_id} {' '.join(parts[1:])}"
                        if new_line != line.strip():
                            file_modified += 1
                        fixed.append(new_line)
                        continue
                    except ValueError:
                        pass
                fixed.append(line)

            if file_modified:
                lbl_path.write_text("\n".join(fixed) + "\n", encoding="utf-8")
                total_fixed_lines += file_modified
                total_files_touched += 1

    logger.info(f"  Lines normalised: {total_fixed_lines} in {total_files_touched} files")
    return total_fixed_lines


def step_count_classes() -> dict[str, dict[str, int]]:
    """Count annotation instances per class per split."""
    logger.info(f"{BANNER}")
    logger.info("STEP 3 — Count class distribution")
    logger.info(BANNER)

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        lbl_dir = LABELS_DIR / split
        split_counts: dict[str, int] = {name: 0 for name in CLASS_NAMES}

        if not lbl_dir.exists():
            counts[split] = split_counts
            continue

        for lbl_path in lbl_dir.glob("*.txt"):
            try:
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(float(parts[0]))
                        if 0 <= cls_id < NUM_CLASSES:
                            split_counts[CLASS_NAMES[cls_id]] += 1
            except Exception:
                continue

        counts[split] = split_counts
        total = sum(split_counts.values())
        logger.info(f"  [{split.upper()}] total annotations: {total}")
        for cls, cnt in split_counts.items():
            logger.info(f"    {cls}: {cnt}")

    return counts


def step_validate_pairs() -> bool:
    """Run the dedicated pair validator."""
    logger.info(f"{BANNER}")
    logger.info("STEP 4 — Validate image-label pairs")
    logger.info(BANNER)

    issues = 0
    for split in SPLITS:
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split

        if not img_dir.exists():
            logger.warning(f"  [{split.upper()}] Image directory not found — skipping")
            continue

        img_stems = {
            f.stem for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        }
        lbl_stems = {
            f.stem for f in lbl_dir.iterdir()
            if f.is_file() and f.suffix == ".txt"
        } if lbl_dir.exists() else set()

        missing_lbls = img_stems - lbl_stems
        missing_imgs = lbl_stems - img_stems

        if missing_lbls:
            logger.warning(f"  [{split.upper()}] {len(missing_lbls)} images have no label")
            issues += len(missing_lbls)
        if missing_imgs:
            logger.warning(f"  [{split.upper()}] {len(missing_imgs)} labels have no image")
            issues += len(missing_imgs)

        if not missing_lbls and not missing_imgs:
            logger.info(f"  [{split.upper()}] All {len(img_stems)} pairs matched ✓")

    return issues == 0


def step_write_data_yaml(counts: dict[str, dict[str, int]]) -> None:
    """Write the final data.yaml with correct absolute path."""
    logger.info(f"{BANNER}")
    logger.info("STEP 5 — Write final data.yaml")
    logger.info(BANNER)

    train_total = sum(counts.get("train", {}).values())
    val_total   = sum(counts.get("val",   {}).values())
    test_total  = sum(counts.get("test",  {}).values())

    yaml_content = f"""# YOLOv8 Dataset Configuration — NEU Metal Surface Defects
# Auto-generated by build_final_dataset.py on {datetime.now().strftime('%Y-%m-%d')}
#
# Train annotations : {train_total}
# Val   annotations : {val_total}
# Test  annotations : {test_total}

path: {YOLO_ROOT.as_posix()}  # absolute dataset root
train: images/train
val:   images/val
test:  images/test

# Number of classes
nc: {NUM_CLASSES}

# Class names (index == YOLO label class ID)
names:
"""
    for idx, name in DEFECT_CLASSES.items():
        yaml_content += f"  {idx}: {name}\n"

    DATA_YAML.write_text(yaml_content, encoding="utf-8")
    logger.info(f"  Written: {DATA_YAML}")


def step_generate_summary_report(counts: dict[str, dict[str, int]],
                                  fixed_lines: int) -> None:
    """Save a JSON + Markdown final dataset summary."""
    logger.info(f"{BANNER}")
    logger.info("STEP 6 — Generate final dataset summary report")
    logger.info(BANNER)

    ensure_dirs(REPORTS_DIR)

    # Image counts per split
    image_counts = {}
    for split in SPLITS:
        img_dir = IMAGES_DIR / split
        image_counts[split] = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

    summary = {
        "generated": datetime.now().isoformat(),
        "dataset_root": str(YOLO_ROOT),
        "float_class_ids_fixed": fixed_lines,
        "image_counts": image_counts,
        "annotation_counts": counts,
        "total_images": sum(image_counts.values()),
        "total_annotations": sum(
            v for split_cnt in counts.values() for v in split_cnt.values()
        ),
    }

    json_path = REPORTS_DIR / "final_dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"  JSON  : {json_path}")

    # Markdown
    md_lines = [
        "# Final Dataset Summary",
        "",
        f"> Generated: {summary['generated']}",
        "",
        "## Image Counts",
        "",
        "| Split | Images |",
        "|-------|--------|",
    ]
    for split, cnt in image_counts.items():
        md_lines.append(f"| {split} | {cnt} |")
    md_lines.append(f"| **Total** | **{summary['total_images']}** |")

    md_lines += ["", "## Annotation Counts per Class", ""]
    header = "| Class |" + "".join(f" {s.capitalize()} |" for s in SPLITS) + " Total |"
    sep    = "|-------|" + "".join("--------|" for _ in SPLITS) + "-------|"
    md_lines += [header, sep]
    for cls in CLASS_NAMES:
        row = f"| {cls} |"
        row_total = 0
        for split in SPLITS:
            cnt = counts.get(split, {}).get(cls, 0)
            row += f" {cnt} |"
            row_total += cnt
        row += f" {row_total} |"
        md_lines.append(row)

    md_lines += [
        "",
        f"**Total annotations:** {summary['total_annotations']}",
        f"**Float class IDs normalised:** {fixed_lines} lines",
        "",
        "---",
        "*Generated by `build_final_dataset.py`*",
    ]

    md_path = REPORTS_DIR / "final_dataset_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info(f"  Markdown: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build & finalize the NEU YOLO dataset pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only; skip augmentation and file writes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    logger.info(BANNER)
    logger.info("REAL-TIME INDUSTRIAL DEFECT DETECTION — DATASET BUILDER")
    logger.info(f"  Mode        : {'DRY RUN' if args.dry_run else 'FULL BUILD'}")
    logger.info(f"  Dataset root: {YOLO_ROOT}")
    logger.info(f"  Timestamp   : {datetime.now().isoformat()}")
    logger.info(BANNER)

    # 1. Verify structure
    structure_ok = step_verify_structure()
    if not structure_ok:
        logger.error("Required directories missing — aborting.")
        sys.exit(1)

    # 2. Fix float class IDs (unless dry-run)
    fixed_lines = 0
    if not args.dry_run:
        fixed_lines = step_fix_float_class_ids()

    # 3. Count class distribution
    counts = step_count_classes()

    # 4. Validate pairs
    pairs_ok = step_validate_pairs()

    # 5. Write data.yaml (unless dry-run)
    if not args.dry_run:
        step_write_data_yaml(counts)

    # 6. Summary report (unless dry-run)
    if not args.dry_run:
        step_generate_summary_report(counts, fixed_lines)

    elapsed = time.time() - start
    status = "✓ COMPLETE" if pairs_ok else "⚠ COMPLETE WITH WARNINGS"
    logger.info(BANNER)
    logger.info(f"PIPELINE {status} in {elapsed:.1f}s")
    logger.info(BANNER)

    sys.exit(0 if pairs_ok else 1)


if __name__ == "__main__":
    main()
