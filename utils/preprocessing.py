"""
Image Preprocessing Utilities
=============================
Real-Time Industrial Defect Detection System

Modular, reusable functions for checking, resizing, normalising,
and saving images for YOLO defect detection.

Optimized on: 2026-07-12 by saniyamirjanavar-hash
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("preprocessing")


def read_and_validate_image(img_path: Path) -> np.ndarray | None:
    """
    Read an image using OpenCV and check for file corruption or dimension issues.
    
    Args:
        img_path: Path to the image file.
        
    Returns:
        The decoded image as a NumPy array, or None if corrupted.
    """
    try:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imread returned None")
        if img.shape[0] == 0 or img.shape[1] == 0:
            raise ValueError("Image has zero dimensions")
        if img.size == 0:
            raise ValueError("Empty image data")
        return img
    except Exception as exc:
        logger.warning("Image validation failed for %s: %s", img_path.name, exc)
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to the specified width and height.
    
    Args:
        image: Input image array.
        target_size: Desired width and height as a tuple (width, height).
        interpolation: OpenCV interpolation mode.
        
    Returns:
        Resized image array.
    """
    h, w = image.shape[:2]
    if (w, h) == target_size:
        return image
    return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image_pixels(
    image: np.ndarray,
    norm_type: str = "min_max",
    mean: List[float] | None = None,
    std: List[float] | None = None
) -> np.ndarray:
    """
    Normalise pixel values using min-max scaling or z-score standardization.
    
    Args:
        image: BGR input image.
        norm_type: "min_max", "imagenet", or "standard".
        mean: Mean values per channel.
        std: Std values per channel.
        
    Returns:
        Normalised float32 image array.
    """
    # Convert BGR to RGB if z-score standardization is applied (which is typically RGB-based)
    if norm_type in ("imagenet", "standard"):
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
    # 1. Base min-max scaling to [0.0, 1.0]
    img = image.astype(np.float32) / 255.0

    if norm_type in ("imagenet", "standard"):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        
        if img.ndim == 3 and img.shape[2] == 3:
            # Inline fast z-score subtraction and division
            np.subtract(img, mean_arr, out=img)
            np.divide(img, std_arr, out=img)

    return img


def preprocess_and_save(
    img_path: Path,
    output_path: Path,
    target_size: Tuple[int, int],
    normalize_config: Dict[str, Any],
    save_format: str,
    interpolation_mode: int = cv2.INTER_LINEAR
) -> bool:
    """
    Load, resize, optionally normalize, and save an image.
    
    Args:
        img_path: Source image file path.
        output_path: Destination file path.
        target_size: (width, height) target dimensions.
        normalize_config: Dictionary with 'enabled', 'type', 'mean', and 'std'.
        save_format: Target suffix (e.g. '.png', '.npy', '.jpg').
        interpolation_mode: OpenCV interpolation identifier.
        
    Returns:
        True if the image was processed and saved successfully, False otherwise.
    """
    img = read_and_validate_image(img_path)
    if img is None:
        return False
        
    img = resize_image(img, target_size, interpolation=interpolation_mode)
    
    # Process normalization
    norm_enabled = normalize_config.get("enabled", True)
    norm_type = normalize_config.get("type", "min_max")
    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])

    if norm_enabled:
        img = normalize_image_pixels(img, norm_type, mean, std)
        
    try:
        # If float32 normalization was applied and saving to .png/jpg, we need to convert back
        # or use .npy for true floating point arrays.
        if save_format == ".npy":
            np.save(str(output_path.with_suffix(".npy")), img)
        else:
            if img.dtype == np.float32:
                # Denormalize for image display if requested format is image file
                img = (img * 255.0)
                img = np.clip(img, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_path), img)
        return True
    except Exception as exc:
        logger.error("Failed to save processed image %s: %s", output_path.name, exc)
        return False
