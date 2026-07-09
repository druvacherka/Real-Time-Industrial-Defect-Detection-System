"""
Image Preprocessing Utilities
=============================
Modular, reusable functions for checking, resizing, normalising,
and saving images for YOLO defect detection.
"""

from __future__ import annotations

import logging
import time
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


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to the specified width and height.
    
    Args:
        image: Input image array.
        target_size: Desired width and height as a tuple (width, height).
        
    Returns:
        Resized image array.
    """
    h, w = image.shape[:2]
    if (w, h) == target_size:
        return image
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image_pixels(image: np.ndarray) -> np.ndarray:
    """
    Normalise pixel values from uint8 [0, 255] to float32 [0.0, 1.0].
    
    Args:
        image: BGR or Grayscale input image.
        
    Returns:
        Normalised float32 image array.
    """
    return image.astype(np.float32) / 255.0


def preprocess_and_save(
    img_path: Path,
    output_path: Path,
    target_size: Tuple[int, int],
    normalize: bool,
    save_format: str
) -> bool:
    """
    Load, resize, optionally normalize, and save an image.
    
    Args:
        img_path: Source image file path.
        output_path: Destination file path.
        target_size: (width, height) target dimensions.
        normalize: If True, normalize pixel values to [0.0, 1.0].
        save_format: Target suffix (e.g. '.png', '.npy', '.jpg').
        
    Returns:
        True if the image was processed and saved successfully, False otherwise.
    """
    img = read_and_validate_image(img_path)
    if img is None:
        return False
        
    img = resize_image(img, target_size)
    
    if normalize and save_format == ".npy":
        img = normalize_image_pixels(img)
        
    try:
        if save_format == ".npy":
            np.save(str(output_path.with_suffix(".npy")), img)
        else:
            if img.dtype == np.float32:
                img = (img * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), img)
        return True
    except Exception as exc:
        logger.error("Failed to save processed image %s: %s", output_path.name, exc)
        return False
