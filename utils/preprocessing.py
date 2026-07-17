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


def resize_image(image: np.ndarray, target_size: Tuple[int, int], interpolation: int = None) -> np.ndarray:
    """
    Resize image to the specified width and height.
    Automatically chooses the best interpolation algorithm if not specified.
    
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
        
    if interpolation is None:
        # Downscaling: INTER_AREA is best to avoid aliasing
        if target_size[0] < w or target_size[1] < h:
            interpolation = cv2.INTER_AREA
        # Upscaling: INTER_CUBIC is slower but higher quality than linear
        else:
            interpolation = cv2.INTER_CUBIC

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
    # Check for empty or invalid image array
    if image is None or image.size == 0:
        raise ValueError("Cannot normalize empty image array")

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
        
        # Ensure std values are positive to avoid division by zero
        if np.any(std_arr <= 0):
            logger.warning("Standard deviation values contain zero or negative numbers. Falling back to 1.0.")
            std_arr = np.where(std_arr <= 0, 1.0, std_arr).astype(np.float32)

        if img.ndim == 3 and img.shape[2] == 3:
            # Inline fast z-score subtraction and division
            np.subtract(img, mean_arr, out=img)
            np.divide(img, std_arr, out=img)

    # Sanity check for NaNs or Infs
    if not np.isfinite(img).all():
        logger.error("Normalized image contains NaN or Inf values. Replacing with zeros.")
        np.nan_to_num(img, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    return img


def preprocess_and_save(
    img_path: Path,
    output_path: Path,
    target_size: Tuple[int, int],
    normalize_config: Dict[str, Any],
    save_format: str,
    interpolation_mode: int = cv2.INTER_LINEAR
) -> Tuple[bool, Dict[str, float]]:
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
        A tuple of (success_status, metrics_dict).
    """
    import time
    metrics = {
        "read_time": 0.0,
        "resize_time": 0.0,
        "normalize_time": 0.0,
        "save_time": 0.0,
        "total_time": 0.0,
        "post_mean": 0.0,
        "post_std": 0.0
    }
    
    t0 = time.time()
    img = read_and_validate_image(img_path)
    metrics["read_time"] = time.time() - t0
    
    if img is None:
        logger.warning(f"Failed to read image at {img_path}")
        metrics["total_time"] = time.time() - t0
        return False, metrics
        
    t1 = time.time()
    try:
        img = resize_image(img, target_size, interpolation=interpolation_mode)
    except Exception as exc:
        logger.error(f"Failed to resize image {img_path.name}: {exc}")
        metrics["total_time"] = time.time() - t0
        return False, metrics
    metrics["resize_time"] = time.time() - t1
    
    # Process normalization
    norm_enabled = normalize_config.get("enabled", True)
    norm_type = normalize_config.get("type", "min_max")
    mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
    std = normalize_config.get("std", [0.229, 0.224, 0.225])

    t2 = time.time()
    try:
        if norm_enabled:
            img = normalize_image_pixels(img, norm_type, mean, std)
    except Exception as exc:
        logger.error(f"Failed to normalize image {img_path.name}: {exc}")
        metrics["total_time"] = time.time() - t0
        return False, metrics
    metrics["normalize_time"] = time.time() - t2
    
    # Generate stats metrics of the preprocessed image
    metrics["post_mean"] = float(np.mean(img))
    metrics["post_std"] = float(np.std(img))
        
    t3 = time.time()
    try:
        # If float32 normalization was applied and saving to .png/jpg, we need to convert back
        # or use .npy for true floating point arrays.
        if save_format == ".npy":
            np.save(str(output_path.with_suffix(".npy")), img)
        else:
            if img.dtype == np.float32:
                # Denormalize for image display if requested format is image file
                if norm_enabled and norm_type in ("imagenet", "standard"):
                    # Bring back from standard format to 0-255
                    mean_arr = np.array(mean, dtype=np.float32)
                    std_arr = np.array(std, dtype=np.float32)
                    img = img * std_arr + mean_arr
                img = (img * 255.0)
                img = np.clip(img, 0, 255).astype(np.uint8)
                if norm_enabled and norm_type in ("imagenet", "standard"):
                    # Convert RGB back to BGR for OpenCV write
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img)
        metrics["save_time"] = time.time() - t3
        metrics["total_time"] = time.time() - t0
        return True, metrics
    except Exception as exc:
        logger.error("Failed to save processed image %s: %s", output_path.name, exc)
        metrics["total_time"] = time.time() - t0
        return False, metrics
