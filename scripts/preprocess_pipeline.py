#!/usr/bin/env python3
"""
Optimized Image Preprocessing Pipeline
======================================
Real-Time Industrial Defect Detection System

Production pipeline utilizing ProcessPoolExecutor to perform parallelized:
  - Configurable resizing of images to uniform dimensions
  - Flexible pixel normalization strategies (min-max, ImageNet, etc.)
  - Verification and filtering of corrupted files
  - Saving processed outputs (supporting .png or .npy formats)
  - Detailed performance metrics and statistics generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
import cv2

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
    IMAGE_EXTENSIONS,
    ensure_dirs,
)
from utils.preprocessing import preprocess_and_save

# Setup report & logs dirs
ensure_dirs(LOGS_DIR, REPORTS_DIR)

# Initialize logging
logger = logging.getLogger("preprocessing_pipeline")
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

# Configuration mapping for OpenCV interpolation
INTERPOLATION_MAP = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def load_preprocessing_config() -> Dict[str, Any]:
    """
    Load settings from configs/preprocessing.yaml.
    """
    config_path = _PROJECT_ROOT / "configs" / "preprocessing.yaml"
    default_config = {
        "target_size": [640, 640],
        "interpolation": "linear",
        "normalization": {
            "enabled": True,
            "type": "min_max",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "num_workers": 4,
        "save_format": ".png"
    }
    
    if not config_path.exists():
        logger.warning("configs/preprocessing.yaml not found. Using default configurations.")
        return default_config
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            # Merge dictionary safely
            for k, v in user_config.items():
                if isinstance(v, dict) and k in default_config:
                    default_config[k].update(v)  # type: ignore
                else:
                    default_config[k] = v  # type: ignore
    except Exception as exc:
        logger.error("Failed to load configs/preprocessing.yaml: %s. Falling back to defaults.", exc)
        
    return default_config


def process_single_image_worker(
    img_path: Path,
    out_path: Path,
    target_size: Tuple[int, int],
    normalize_config: Dict[str, Any],
    save_format: str,
    interpolation_mode: int
) -> Tuple[str, bool, Dict[str, float]]:
    """
    Worker function to process a single image. Returns (filename, success, metrics_dict).
    """
    success, metrics = preprocess_and_save(
        img_path,
        out_path,
        target_size,
        normalize_config,
        save_format,
        interpolation_mode
    )
    return img_path.name, success, metrics


class PreprocessingPipeline:
    """Multi-processed image preprocessing pipeline coordinator."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.input_dir = INPUT_DIR
        self.output_dir = OUTPUT_DIR
        
        # Load params from config
        self.target_size = tuple(self.config.get("target_size", [640, 640]))
        self.interpolation_str = self.config.get("interpolation", "linear")
        self.interpolation_mode = INTERPOLATION_MAP.get(self.interpolation_str, cv2.INTER_LINEAR)
        self.normalize_config = self.config.get("normalization", {"enabled": True, "type": "min_max"})
        self.save_format = self.config.get("save_format", ".png")
        self.num_workers = int(self.config.get("num_workers", 4))

        # Statistics tracker
        self.stats = {
            "total_processed": 0,
            "total_skipped": 0,
            "per_split": {},
            "processing_times": [],
            "read_times": [],
            "resize_times": [],
            "normalize_times": [],
            "save_times": [],
        }

    def setup_dirs(self) -> None:
        """Ensure output directories exist."""
        for split in SPLITS:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

    def process_split(self, split: str) -> None:
        """Process all images in a given split directory using process pool."""
        in_split_dir = self.input_dir / split
        out_split_dir = self.output_dir / split

        if not in_split_dir.exists():
            logger.info("  [%s] Input split directory not found, skipping", split.upper())
            return

        img_files = sorted([
            f for f in in_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not img_files:
            logger.info("  [%s] No images found to preprocess", split.upper())
            return

        processed = 0
        skipped = 0
        
        logger.info("  [%s] Submitting %d images to ProcessPoolExecutor (workers=%d)...", 
                    split.upper(), len(img_files), self.num_workers)

        # Prepare worker task parameters
        tasks = []
        for img_path in img_files:
            out_path = out_split_dir / f"{img_path.stem}{self.save_format}"
            tasks.append((
                img_path,
                out_path,
                self.target_size,
                self.normalize_config,
                self.save_format,
                self.interpolation_mode
            ))

        # Run concurrently
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_single_image_worker, *t) for t in tasks]
            
            for future in futures:
                try:
                    name, success, metrics = future.result()
                    if success:
                        processed += 1
                        self.stats["processing_times"].append(metrics["total_time"])
                        self.stats["read_times"].append(metrics["read_time"])
                        self.stats["resize_times"].append(metrics["resize_time"])
                        self.stats["normalize_times"].append(metrics["normalize_time"])
                        self.stats["save_times"].append(metrics["save_time"])
                    else:
                        skipped += 1
                except Exception as exc:
                    logger.error("Error occurred in preprocessing worker: %s", exc)
                    skipped += 1

        self.stats["per_split"][split] = {
            "processed": processed,
            "skipped": skipped,
            "total": len(img_files)
        }
        
        logger.info(
            "  [%s] Preprocessing stats — Processed: %d | Skipped: %d / %d",
            split.upper(), processed, skipped, len(img_files)
        )

    def save_stats(self) -> Path:
        """Save pipeline stats to reports/preprocessing_statistics.json."""
        ensure_dirs(REPORTS_DIR)
        report_path = REPORTS_DIR / "preprocessing_statistics.json"
        
        times = self.stats["processing_times"]
        read_t = self.stats["read_times"]
        resize_t = self.stats["resize_times"]
        norm_t = self.stats["normalize_times"]
        save_t = self.stats["save_times"]
        
        summary = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "target_size": list(self.target_size),
            "interpolation": self.interpolation_str,
            "normalization": self.normalize_config,
            "save_format": self.save_format,
            "total_processed": self.stats["total_processed"],
            "total_skipped": self.stats["total_skipped"],
            "per_split": self.stats["per_split"],
            "timing": {
                "mean_ms": round(float(np.mean(times)) * 1000, 2) if times else 0.0,
                "median_ms": round(float(np.median(times)) * 1000, 2) if times else 0.0,
                "total_seconds": round(float(sum(times)), 2) if times else 0.0,
                "stages_mean_ms": {
                    "read": round(float(np.mean(read_t)) * 1000, 2) if read_t else 0.0,
                    "resize": round(float(np.mean(resize_t)) * 1000, 2) if resize_t else 0.0,
                    "normalize": round(float(np.mean(norm_t)) * 1000, 2) if norm_t else 0.0,
                    "save": round(float(np.mean(save_t)) * 1000, 2) if save_t else 0.0,
                }
            }
        }
        
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        return report_path

    def generate_augmentation_summary(self) -> None:
        """
        Generate a markdown summary report of the preprocessing step.
        """
        summary_md_path = REPORTS_DIR / "preprocessing_validation_summary.md"
        
        times = self.stats["processing_times"]
        read_t = self.stats["read_times"]
        resize_t = self.stats["resize_times"]
        norm_t = self.stats["normalize_times"]
        save_t = self.stats["save_times"]
        
        mean_ms = round(float(np.mean(times)) * 1000, 2) if times else 0.0
        total_sec = round(float(sum(times)), 2) if times else 0.0
        
        mean_read = round(float(np.mean(read_t)) * 1000, 2) if read_t else 0.0
        mean_resize = round(float(np.mean(resize_t)) * 1000, 2) if resize_t else 0.0
        mean_norm = round(float(np.mean(norm_t)) * 1000, 2) if norm_t else 0.0
        mean_save = round(float(np.mean(save_t)) * 1000, 2) if save_t else 0.0

        lines = [
            "# Preprocessing Pipeline Run Summary",
            "",
            f"**Execution Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Image Dimensions**: {self.target_size[0]}x{self.target_size[1]} ({self.interpolation_str} interpolation)",
            f"**Normalization Mode**: {self.normalize_config.get('type', 'min_max')} (enabled={self.normalize_config.get('enabled', True)})",
            f"**Output Format**: `{self.save_format}`",
            "",
            "## 1. Processed Split Counts",
            "",
            "| Split | Total Input | Preprocessed Successfully | Skipped (Corrupt) |",
            "| --- | --- | --- | --- |",
        ]
        
        for split, metrics in self.stats["per_split"].items():
            lines.append(f"| {split.capitalize()} | {metrics['total']} | {metrics['processed']} | {metrics['skipped']} |")
            
        lines.append("")
        
        lines.extend([
            "## 2. Performance Summary",
            "",
            f"- **Concurrency Level**: {self.num_workers} Parallel Processes",
            f"- **Mean Processing Speed**: {mean_ms} ms per image",
            f"- **Total Time Spent**: {total_sec} seconds",
            "",
            "### Stage Timing Breakdown (Mean):",
            f"- **Read & Validate**: {mean_read} ms",
            f"- **Resize**: {mean_resize} ms",
            f"- **Normalization**: {mean_norm} ms",
            f"- **Save & Format**: {mean_save} ms",
            "",
            "---",
            "*Report generated automatically by `preprocess_pipeline.py`.*"
        ])

        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Successfully generated preprocessing validation summary markdown: %s", summary_md_path)

    def run(self) -> None:
        """Run full preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING OPTIMIZED PREPROCESSING PIPELINE (CONCURRENT)")
        logger.info("=" * 60)

        start_time = time.time()
        self.setup_dirs()

        for split in SPLITS:
            self.process_split(split)

        self.stats["total_processed"] = sum(s["processed"] for s in self.stats["per_split"].values())
        self.stats["total_skipped"] = sum(s["skipped"] for s in self.stats["per_split"].values())

        report_path = self.save_stats()
        self.generate_augmentation_summary()
        
        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("OPTIMIZED PREPROCESSING PIPELINE COMPLETE in %.2fs", elapsed)
        logger.info("  Processed: %d", self.stats["total_processed"])
        logger.info("  Skipped:   %d", self.stats["total_skipped"])
        logger.info("  Stats Report: %s", report_path)
        logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optimized Parallel Preprocessing Pipeline.")
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers (processes) to run concurrently."
    )
    args = parser.parse_args()

    config = load_preprocessing_config()
    if args.workers:
        config["num_workers"] = args.workers

    pipeline = PreprocessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
