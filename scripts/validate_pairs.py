#!/usr/bin/env python3
"""
Image-Label Pair Validator
===========================
Real-Time Industrial Defect Detection System

Dedicated script to validate every image-label pair in the YOLO dataset:
  - Checks image readability (via OpenCV)
  - Validates YOLO label format (5 fields per line)
  - Checks class ID range [0, NUM_CLASSES)
  - Checks bbox coordinates in [0, 1]
  - Checks positive width / height
  - Detects orphaned images (no label) and orphaned labels (no image)
  - Detects empty annotation files
  - Outputs JSON + Markdown reports

Exit codes:
  0 — all pairs valid (or only warnings)
  1 — critical errors found

Author: saniyamirjanavar-hash
Date: 2026-07-07
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Allow running as a standalone script from the project root
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from config import (
    IMAGES_DIR, LABELS_DIR, REPORTS_DIR, LOGS_DIR,
    SPLITS, NUM_CLASSES, IMAGE_EXTENSIONS, LABEL_EXTENSION,
    ensure_dirs,
)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
ensure_dirs(LOGS_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "validate_pairs.log", mode="a"),
    ],
)
logger = logging.getLogger("validate_pairs")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class PairValidator:
    """Validate all image-label pairs across every dataset split."""

    def __init__(self, images_dir: Path = IMAGES_DIR, labels_dir: Path = LABELS_DIR):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.timestamp = datetime.now().isoformat()
        self.report: dict = {
            "timestamp": self.timestamp,
            "overall_status": "PASS",
            "splits": {},
            "summary": {
                "total_images": 0,
                "total_labels": 0,
                "orphaned_images": 0,
                "orphaned_labels": 0,
                "empty_label_files": 0,
                "unreadable_images": 0,
                "invalid_bbox_lines": 0,
                "critical_errors": 0,
            },
        }

    # ------------------------------------------------------------------
    # Per-file checks
    # ------------------------------------------------------------------

    def _check_image(self, img_path: Path) -> list[str]:
        """Return list of error strings for a single image file."""
        errors: list[str] = []
        if not HAS_OPENCV:
            return errors  # skip if OpenCV unavailable
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"[UNREADABLE] {img_path.name}: cv2.imread returned None")
            elif img.shape[0] == 0 or img.shape[1] == 0:
                errors.append(f"[ZERO-DIM] {img_path.name}: dimensions {img.shape}")
        except Exception as exc:
            errors.append(f"[ERROR] {img_path.name}: {exc}")
        return errors

    def _check_label(self, lbl_path: Path) -> list[str]:
        """Return list of error strings for a single label file."""
        errors: list[str] = []
        if not lbl_path.exists():
            errors.append(f"[MISSING] {lbl_path.name}: file does not exist")
            return errors

        try:
            lines = lbl_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            errors.append(f"[UNREADABLE] {lbl_path.name}: {exc}")
            return errors

        non_empty = [l.strip() for l in lines if l.strip()]
        if not non_empty:
            errors.append(f"[EMPTY] {lbl_path.name}: no annotation lines")
            return errors

        for i, line in enumerate(non_empty, 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(
                    f"[FORMAT] {lbl_path.name}:L{i}: "
                    f"expected 5 fields, got {len(parts)} → '{line}'"
                )
                continue

            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = (float(x) for x in parts[1:])
            except ValueError:
                errors.append(
                    f"[PARSE] {lbl_path.name}:L{i}: non-numeric values → '{line}'"
                )
                continue

            if cls_id < 0 or cls_id >= NUM_CLASSES:
                errors.append(
                    f"[CLS_ID] {lbl_path.name}:L{i}: "
                    f"class {cls_id} out of [0, {NUM_CLASSES - 1}]"
                )

            for name, val in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
                if val < 0 or val > 1:
                    errors.append(
                        f"[RANGE] {lbl_path.name}:L{i}: "
                        f"{name}={val:.4f} outside [0, 1]"
                    )

            if w <= 0:
                errors.append(f"[DIM] {lbl_path.name}:L{i}: width={w} ≤ 0")
            if h <= 0:
                errors.append(f"[DIM] {lbl_path.name}:L{i}: height={h} ≤ 0")

        return errors

    # ------------------------------------------------------------------
    # Per-split validation
    # ------------------------------------------------------------------

    def validate_split(self, split: str) -> dict:
        """Validate all pairs in one split. Returns per-split stats."""
        img_dir = self.images_dir / split
        lbl_dir = self.labels_dir / split

        split_result: dict = {
            "images": 0,
            "labels": 0,
            "orphaned_images": [],
            "orphaned_labels": [],
            "empty_label_files": [],
            "image_errors": [],
            "label_errors": [],
        }

        if not img_dir.exists() and not lbl_dir.exists():
            logger.warning(f"  [{split.upper()}] Directories not found — skipping")
            return split_result

        img_stems: dict[str, Path] = {}
        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    img_stems[f.stem] = f

        lbl_stems: dict[str, Path] = {}
        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.is_file() and f.suffix == LABEL_EXTENSION:
                    lbl_stems[f.stem] = f

        split_result["images"] = len(img_stems)
        split_result["labels"] = len(lbl_stems)

        # Orphans
        orphan_imgs = sorted(img_stems.keys() - lbl_stems.keys())
        orphan_lbls = sorted(lbl_stems.keys() - img_stems.keys())
        split_result["orphaned_images"] = orphan_imgs[:50]   # cap for report size
        split_result["orphaned_labels"] = orphan_lbls[:50]

        if orphan_imgs:
            logger.warning(f"  [{split.upper()}] {len(orphan_imgs)} images have no label")
        if orphan_lbls:
            logger.warning(f"  [{split.upper()}] {len(orphan_lbls)} labels have no image")

        # Check matched pairs
        matched_stems = img_stems.keys() & lbl_stems.keys()
        for stem in sorted(matched_stems):
            img_errs = self._check_image(img_stems[stem])
            lbl_errs = self._check_label(lbl_stems[stem])

            if img_errs:
                split_result["image_errors"].extend(img_errs[:5])
            if lbl_errs:
                # Classify empty label separately
                if any("[EMPTY]" in e for e in lbl_errs):
                    split_result["empty_label_files"].append(lbl_stems[stem].name)
                else:
                    split_result["label_errors"].extend(lbl_errs[:5])

        total_img_errs = len(split_result["image_errors"])
        total_lbl_errs = len(split_result["label_errors"])
        total_empty = len(split_result["empty_label_files"])

        logger.info(
            f"  [{split.upper()}] Images: {split_result['images']} | "
            f"Labels: {split_result['labels']} | "
            f"Orphan imgs: {len(orphan_imgs)} | "
            f"Orphan lbls: {len(orphan_lbls)} | "
            f"Empty labels: {total_empty} | "
            f"Img errors: {total_img_errs} | "
            f"Lbl errors: {total_lbl_errs}"
        )

        return split_result

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _build_markdown(self) -> str:
        """Render the report as Markdown."""
        status_icon = {
            "PASS": "✅", "WARN": "⚠️", "FAIL": "❌"
        }.get(self.report["overall_status"], "❓")

        s = self.report["summary"]
        lines = [
            "# Image-Label Pair Validation Report",
            "",
            f"> Generated: {self.timestamp}",
            f"> Status: {status_icon} **{self.report['overall_status']}**",
            "",
            "## Summary",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Total images | {s['total_images']} |",
            f"| Total labels | {s['total_labels']} |",
            f"| Orphaned images (no label) | {s['orphaned_images']} |",
            f"| Orphaned labels (no image) | {s['orphaned_labels']} |",
            f"| Empty label files | {s['empty_label_files']} |",
            f"| Unreadable images | {s['unreadable_images']} |",
            f"| Invalid bbox lines | {s['invalid_bbox_lines']} |",
            f"| Critical errors | {s['critical_errors']} |",
            "",
        ]

        for split, data in self.report["splits"].items():
            lines += [
                f"## {split.capitalize()} Split",
                "",
                f"| Images | Labels | Orphan imgs | Orphan lbls | Empty lbls |",
                f"|--------|--------|-------------|-------------|------------|",
                f"| {data['images']} | {data['labels']} | "
                f"{len(data['orphaned_images'])} | "
                f"{len(data['orphaned_labels'])} | "
                f"{len(data['empty_label_files'])} |",
                "",
            ]
            if data["image_errors"]:
                lines.append("### Image Errors (first 5)")
                for e in data["image_errors"][:5]:
                    lines.append(f"- `{e}`")
                lines.append("")
            if data["label_errors"]:
                lines.append("### Label Errors (first 5)")
                for e in data["label_errors"][:5]:
                    lines.append(f"- `{e}`")
                lines.append("")

        lines += ["---", "*Generated by `validate_pairs.py`*"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Execute full validation. Returns True if no critical errors."""
        logger.info("=" * 60)
        logger.info("IMAGE-LABEL PAIR VALIDATION")
        logger.info("=" * 60)

        s = self.report["summary"]
        critical = 0

        for split in SPLITS:
            logger.info(f"Validating [{split.upper()}] split …")
            result = self.validate_split(split)
            self.report["splits"][split] = result

            s["total_images"] += result["images"]
            s["total_labels"] += result["labels"]
            s["orphaned_images"] += len(result["orphaned_images"])
            s["orphaned_labels"] += len(result["orphaned_labels"])
            s["empty_label_files"] += len(result["empty_label_files"])
            s["unreadable_images"] += len(result["image_errors"])
            s["invalid_bbox_lines"] += len(result["label_errors"])

            critical += (
                len(result["image_errors"])
                + len(result["label_errors"])
                + len(result["orphaned_images"])
                + len(result["orphaned_labels"])
            )

        s["critical_errors"] = critical

        if critical > 0:
            self.report["overall_status"] = "FAIL"
        elif s["empty_label_files"] > 0:
            self.report["overall_status"] = "WARN"

        # Save reports
        ensure_dirs(REPORTS_DIR)
        json_path = REPORTS_DIR / "validation_report.json"
        md_path = REPORTS_DIR / "validation_report.md"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.report, fh, indent=2, default=str)
        md_path.write_text(self._build_markdown(), encoding="utf-8")

        logger.info("=" * 60)
        logger.info(f"VALIDATION COMPLETE — Status: {self.report['overall_status']}")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  MD:   {md_path}")
        logger.info("=" * 60)

        return self.report["overall_status"] != "FAIL"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    validator = PairValidator()
    success = validator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
