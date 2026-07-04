"""
Data Loader Utilities
=====================
Provides helper functions for loading images and annotations
from the NEU Metal Surface Defects dataset.

Supports multiple image formats and Pascal VOC XML annotations.

Author: druvacherka
Date: 2026-07-04
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

from utils.constants import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_ANNOTATION_FORMAT,
    CLASS_NAMES,
)
from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def discover_images(directory: str) -> List[str]:
    """
    Recursively discover all supported image files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of absolute image paths.
    """
    image_paths: List[str] = []
    for fmt in SUPPORTED_IMAGE_FORMATS:
        pattern = os.path.join(directory, "**", f"*{fmt}")
        image_paths.extend(glob.glob(pattern, recursive=True))
    image_paths.sort()
    logger.info("Discovered %d images in %s", len(image_paths), directory)
    return image_paths


def discover_annotations(directory: str) -> List[str]:
    """
    Recursively discover all annotation XML files in a directory.

    Args:
        directory: Root directory to search.

    Returns:
        Sorted list of absolute annotation paths.
    """
    pattern = os.path.join(directory, "**", f"*{SUPPORTED_ANNOTATION_FORMAT}")
    annotation_paths = sorted(glob.glob(pattern, recursive=True))
    logger.info("Discovered %d annotations in %s", len(annotation_paths), directory)
    return annotation_paths


def load_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    color_mode: str = "bgr",
) -> Optional[np.ndarray]:
    """
    Load a single image from disk.

    Args:
        image_path: Path to the image file.
        target_size: Optional (width, height) to resize.
        color_mode: 'bgr' (default) or 'rgb'.

    Returns:
        Numpy array of the image or None if loading failed.
    """
    if not os.path.isfile(image_path):
        logger.warning("Image file not found: %s", image_path)
        return None

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Failed to decode image: %s", image_path)
        return None

    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    return image


def load_image_batch(
    image_paths: List[str],
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Load a batch of images.

    Args:
        image_paths: List of image file paths.
        target_size: Optional resize dimensions.

    Returns:
        List of successfully loaded images.
    """
    images: List[np.ndarray] = []
    for path in image_paths:
        img = load_image(path, target_size=target_size)
        if img is not None:
            images.append(img)
    logger.info("Loaded %d / %d images", len(images), len(image_paths))
    return images


def pair_images_and_annotations(
    image_dir: str,
    annotation_dir: str,
) -> List[Dict[str, str]]:
    """
    Pair image files with their corresponding annotation files by stem name.

    Args:
        image_dir: Directory containing images.
        annotation_dir: Directory containing XML annotations.

    Returns:
        List of dicts with keys 'image' and 'annotation'.
    """
    images = discover_images(image_dir)
    annotations = discover_annotations(annotation_dir)

    annotation_map: Dict[str, str] = {}
    for ann_path in annotations:
        stem = Path(ann_path).stem
        annotation_map[stem] = ann_path

    pairs: List[Dict[str, str]] = []
    missing_annotations: List[str] = []

    for img_path in images:
        stem = Path(img_path).stem
        if stem in annotation_map:
            pairs.append({"image": img_path, "annotation": annotation_map[stem]})
        else:
            missing_annotations.append(img_path)

    if missing_annotations:
        logger.warning(
            "%d images have no matching annotation", len(missing_annotations)
        )

    logger.info("Paired %d image-annotation entries", len(pairs))
    return pairs


def get_class_index(class_name: str) -> int:
    """
    Return the integer index for a defect class name.

    Args:
        class_name: Name of the defect class.

    Returns:
        Zero-based class index.

    Raises:
        ValueError: If the class name is not recognized.
    """
    normalized = class_name.strip().lower()
    for idx, name in enumerate(CLASS_NAMES):
        if name.lower() == normalized:
            return idx
    raise ValueError(f"Unknown class name: '{class_name}'")


def compute_image_statistics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute basic pixel statistics for a single image.

    Args:
        image: Numpy array (H, W, C) in uint8.

    Returns:
        Dict with mean, std, min, and max per channel.
    """
    stats: Dict[str, float] = {}
    for i, channel in enumerate(["blue", "green", "red"]):
        ch = image[:, :, i].astype(np.float64)
        stats[f"{channel}_mean"] = float(np.mean(ch))
        stats[f"{channel}_std"] = float(np.std(ch))
        stats[f"{channel}_min"] = float(np.min(ch))
        stats[f"{channel}_max"] = float(np.max(ch))
    return stats
