#!/usr/bin/env python3
"""
Image Preprocessing Pipeline (Refactored)
========================================
Real-Time Industrial Defect Detection System

Production pipeline for:
  - Resizing images to uniform dimensions
  - Normalizing pixel values
  - Skipping corrupted files
  - Saving processed images
  - Generating preprocessing statistics

Utilizes the modular preprocessing helpers in utils.preprocessing.
Generates reports/preprocessing_statistics.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    IMAGES_DIR as INPUT_DIR,
    PROCESSED_DIR as OUTPUT_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    SPLITS,
    TARGET_SIZE,
    IMAGE_EXTENSIONS,
    ensure_dirs,
)
from utils.preprocessing import preprocess_and_save

# Initialize logging
ensure_dirs(LOGS_DIR, REPORTS_DIR)
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)

if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "preprocess_pipeline.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


class PreprocessingPipeline:
    """End-to-end image preprocessing pipeline coordinator."""

    def __init__(
        self,
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        normalize: bool = True,
        save_format: str = ".png",
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = TARGET_SIZE
        self.normalize = normalize
        self.save_format = save_format

        # Statistics tracker
        self.stats = {
            "total_processed": 0,
            "total_skipped": 0,
            "per_split": {},
            "processing_times": [],
        }

    def setup_dirs(self) -> None:
        """Ensure output directories exist."""
        for split in SPLITS:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

    def process_split(self, split: str) -> None:
        """Process all images in a given split directory."""
        in_split_dir = self.input_dir / split
        out_split_dir = self.output_dir / split

        if not in_split_dir.exists():
            logger.info("  [%s] Input split dir not found, skipping", split.upper())
            return

        img_files = sorted([
            f for f in in_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        processed = 0
        skipped = 0

        for img_path in img_files:
            out_path = out_split_dir / f"{img_path.stem}{self.save_format}"
            
            start_time = time.time()
            success = preprocess_and_save(
                img_path,
                out_path,
                self.target_size,
                self.normalize,
                self.save_format
            )
            elapsed = time.time() - start_time
            
            if success:
                processed += 1
                self.stats["processing_times"].append(elapsed)
            else:
                skipped += 1

        self.stats["per_split"][split] = {
            "processed": processed,
            "skipped": skipped,
            "total": len(img_files)
        }
        
        logger.info(
            "  [%s] Processed: %d | Skipped: %d / %d",
            split.upper(), processed, skipped, len(img_files)
        )

    def save_stats(self) -> Path:
        """Save pipeline stats to reports/preprocessing_statistics.json."""
        report_path = REPORTS_DIR / "preprocessing_statistics.json"
        
        times = self.stats["processing_times"]
        summary = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "target_size": list(self.target_size),
            "normalize": self.normalize,
            "save_format": self.save_format,
            "total_processed": self.stats["total_processed"],
            "total_skipped": self.stats["total_skipped"],
            "per_split": self.stats["per_split"],
            "timing": {
                "mean_ms": round(np.mean(times) * 1000, 2) if times else 0,
                "median_ms": round(np.median(times) * 1000, 2) if times else 0,
                "total_seconds": round(sum(times), 2) if times else 0,
            }
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        return report_path

    def run(self) -> None:
        """Run full preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()
        self.setup_dirs()

        for split in SPLITS:
            self.process_split(split)

        self.stats["total_processed"] = sum(s["processed"] for s in self.stats["per_split"].values())
        self.stats["total_skipped"] = sum(s["skipped"] for s in self.stats["per_split"].values())

        report_path = self.save_stats()
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETE in %.2fs", elapsed)
        logger.info("  Processed: %d", self.stats["total_processed"])
        logger.info("  Skipped:   %d", self.stats["total_skipped"])
        logger.info("  Stats Report: %s", report_path)
        logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Preprocessing Pipeline.")
    parser.add_argument(
        "--no-norm",
        action="store_false",
        dest="normalize",
        help="Disable float normalization."
    )
    args = parser.parse_args()

    pipeline = PreprocessingPipeline(normalize=args.normalize)
    pipeline.run()


if __name__ == "__main__":
    main()
