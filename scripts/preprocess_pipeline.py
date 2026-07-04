#!/usr/bin/env python3
"""
Image Preprocessing Pipeline
==============================
Real-Time Industrial Defect Detection System

Production pipeline for:
  - Resizing images to uniform dimensions
  - Normalizing pixel values
  - Removing corrupted files
  - Saving processed images
  - Generating preprocessing statistics
  - Hooks for Albumentations augmentation

Author: saniyamirjanavar-hash
Date: 2026-07-04
"""

import os
import sys
import time
import logging
import json
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Optional, List, Callable

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("[WARNING] OpenCV not installed. Install with: pip install opencv-python")

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
        logging.FileHandler(LOG_DIR / "preprocess_pipeline.log", mode="a"),
    ],
)
logger = logging.getLogger("preprocess_pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
INPUT_DIR = DATASET_DIR / "yolo" / "images"
OUTPUT_DIR = DATASET_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_TARGET_SIZE = (640, 640)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ["train", "val", "test"]


class AugmentationHook:
    """
    Base class for augmentation hooks.

    Subclass this and override `__call__` to integrate Albumentations
    or any other augmentation library in the future.

    Example:
        class AlbumentationsHook(AugmentationHook):
            def __init__(self):
                import albumentations as A
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.2),
                ])

            def __call__(self, image: np.ndarray) -> np.ndarray:
                return self.transform(image=image)["image"]
    """

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation (no-op by default)."""
        return image


class PreprocessingPipeline:
    """
    End-to-end image preprocessing pipeline.

    Steps:
        1. Scan source directories for images
        2. Read and validate each image
        3. Remove / skip corrupted files
        4. Resize to target dimensions
        5. Normalize pixel values to [0, 1]
        6. Apply optional augmentation hook
        7. Save processed images
        8. Generate statistics report
    """

    def __init__(
        self,
        input_dir: Path = INPUT_DIR,
        output_dir: Path = OUTPUT_DIR,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        normalize: bool = True,
        save_format: str = ".png",
        augmentation_hook: Optional[AugmentationHook] = None,
    ):
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV is required for the preprocessing pipeline.")

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.normalize = normalize
        self.save_format = save_format
        self.augmentation_hook = augmentation_hook or AugmentationHook()

        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_skipped": 0,
            "total_corrupted": 0,
            "per_split": {},
            "original_sizes": [],
            "processing_times": [],
        }

        self._corrupted_files: List[str] = []

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    def setup_output_dirs(self):
        """Create output directory structure."""
        for split in SPLITS:
            out_dir = self.output_dir / split
            out_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directories created under {self.output_dir}")

    # ------------------------------------------------------------------
    # Image reading and corruption check
    # ------------------------------------------------------------------
    def read_and_validate(self, img_path: Path) -> Optional[np.ndarray]:
        """
        Read an image and check for corruption.

        Returns None for corrupted files.
        """
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imread returned None")
            if img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Zero-dimension image")
            if img.size == 0:
                raise ValueError("Empty image array")
            return img
        except Exception as exc:
            logger.warning(f"  [CORRUPTED] {img_path.name}: {exc}")
            self._corrupted_files.append(str(img_path))
            self.stats["total_corrupted"] += 1
            return None

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------
    def resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        h, w = img.shape[:2]
        if (w, h) == self.target_size:
            return img
        return cv2.resize(
            img,
            self.target_size,
            interpolation=cv2.INTER_LINEAR,
        )

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_pixels(img: np.ndarray) -> np.ndarray:
        """Normalize pixel values from [0, 255] to [0.0, 1.0]."""
        return img.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Process single image
    # ------------------------------------------------------------------
    def process_image(self, img_path: Path, output_path: Path) -> bool:
        """
        Process a single image through the full pipeline.

        Returns True if successful, False otherwise.
        """
        start = time.time()

        # Read & validate
        img = self.read_and_validate(img_path)
        if img is None:
            return False

        original_h, original_w = img.shape[:2]
        self.stats["original_sizes"].append((original_w, original_h))

        # Resize
        img = self.resize(img)

        # Normalize (save as float32 .npy or convert back for image formats)
        if self.normalize and self.save_format == ".npy":
            img = self.normalize_pixels(img)

        # Augmentation hook
        img = self.augmentation_hook(img)

        # Save
        try:
            if self.save_format == ".npy":
                np.save(str(output_path.with_suffix(".npy")), img)
            else:
                if img.dtype == np.float32:
                    img = (img * 255).astype(np.uint8)
                cv2.imwrite(str(output_path), img)
        except Exception as exc:
            logger.error(f"  Failed to save {output_path.name}: {exc}")
            return False

        elapsed = time.time() - start
        self.stats["processing_times"].append(elapsed)
        return True

    # ------------------------------------------------------------------
    # Process split
    # ------------------------------------------------------------------
    def process_split(self, split: str) -> dict:
        """Process all images in a given split."""
        in_dir = self.input_dir / split
        out_dir = self.output_dir / split

        if not in_dir.exists():
            logger.info(f"  [{split.upper()}] Input directory not found, skipping")
            return {"processed": 0, "skipped": 0, "corrupted": 0}

        image_files = sorted([
            f for f in in_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        processed = 0
        skipped = 0

        for img_path in image_files:
            out_path = out_dir / f"{img_path.stem}{self.save_format}"

            if self.process_image(img_path, out_path):
                processed += 1
            else:
                skipped += 1

        split_stats = {
            "processed": processed,
            "skipped": skipped,
            "total": len(image_files),
        }
        self.stats["per_split"][split] = split_stats

        logger.info(
            f"  [{split.upper()}] Processed: {processed} | "
            f"Skipped: {skipped} / {len(image_files)}"
        )
        return split_stats

    # ------------------------------------------------------------------
    # Generate statistics report
    # ------------------------------------------------------------------
    def generate_statistics(self) -> Path:
        """Generate and save preprocessing statistics."""
        report_path = REPORTS_DIR / "preprocessing_statistics.json"

        times = self.stats["processing_times"]
        sizes = self.stats["original_sizes"]

        summary = {
            "target_size": list(self.target_size),
            "normalize": self.normalize,
            "save_format": self.save_format,
            "total_processed": self.stats["total_processed"],
            "total_skipped": self.stats["total_skipped"],
            "total_corrupted": self.stats["total_corrupted"],
            "corrupted_files": self._corrupted_files,
            "per_split": self.stats["per_split"],
            "timing": {
                "mean_ms": round(np.mean(times) * 1000, 2) if times else 0,
                "median_ms": round(np.median(times) * 1000, 2) if times else 0,
                "max_ms": round(max(times) * 1000, 2) if times else 0,
                "total_seconds": round(sum(times), 2) if times else 0,
            },
            "original_sizes": {
                "unique_sizes": len(set(sizes)),
                "min_width": min(s[0] for s in sizes) if sizes else 0,
                "max_width": max(s[0] for s in sizes) if sizes else 0,
                "min_height": min(s[1] for s in sizes) if sizes else 0,
                "max_height": max(s[1] for s in sizes) if sizes else 0,
            },
        }

        with open(report_path, "w") as fh:
            json.dump(summary, fh, indent=2)

        logger.info(f"Statistics saved to {report_path}")
        return report_path

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info(f"  Input:  {self.input_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Target: {self.target_size}")
        logger.info(f"  Normalize: {self.normalize}")
        logger.info("=" * 60)

        pipeline_start = time.time()
        self.setup_output_dirs()

        for split in SPLITS:
            self.process_split(split)

        self.stats["total_processed"] = sum(
            s.get("processed", 0) for s in self.stats["per_split"].values()
        )
        self.stats["total_skipped"] = sum(
            s.get("skipped", 0) for s in self.stats["per_split"].values()
        )

        report = self.generate_statistics()

        elapsed = time.time() - pipeline_start
        logger.info("=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"  Processed: {self.stats['total_processed']}")
        logger.info(f"  Skipped:   {self.stats['total_skipped']}")
        logger.info(f"  Corrupted: {self.stats['total_corrupted']}")
        logger.info(f"  Time:      {elapsed:.2f}s")
        logger.info(f"  Report:    {report}")
        logger.info("=" * 60)


def main():
    """Entry point for the preprocessing pipeline."""
    pipeline = PreprocessingPipeline(
        target_size=DEFAULT_TARGET_SIZE,
        normalize=True,
        save_format=".png",
    )
    pipeline.run()


if __name__ == "__main__":
    main()
