"""
Image Preprocessing Service
============================
Handles image reading, validation, color conversion, and resizing
for the defect detection inference pipeline.

Author: prajwaledu802-coder
Date: 2026-07-04
"""

import logging
from typing import Tuple, Optional

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

logger = logging.getLogger("defect_detection.image_service")

# Default YOLO input dimensions
DEFAULT_INPUT_SIZE = (640, 640)
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 8192
SUPPORTED_CHANNELS = {1, 3, 4}


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


class ImagePreprocessingService:
    """
    Production-ready image preprocessing service for YOLOv8 inference.

    Pipeline:
        1. Read image from file path or bytes
        2. Validate dimensions and channels
        3. Convert BGR to RGB color space
        4. Resize to model input dimensions
        5. Normalize pixel values to [0, 1]
        6. Return processed NumPy array
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        normalize: bool = True,
        interpolation: int = None,
    ):
        if not HAS_OPENCV:
            raise RuntimeError(
                "OpenCV is required for ImagePreprocessingService. "
                "Install it with: pip install opencv-python"
            )

        self.target_size = target_size
        self.normalize = normalize
        self.interpolation = interpolation or cv2.INTER_LINEAR
        logger.info(
            f"ImagePreprocessingService initialized: "
            f"target={target_size}, normalize={normalize}"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def read_image(self, file_path: str) -> np.ndarray:
        """
        Read an image from disk using OpenCV.

        Args:
            file_path: Absolute or relative path to the image file.

        Returns:
            BGR image as a NumPy array.

        Raises:
            ImageValidationError: If the file cannot be read.
        """
        logger.debug(f"Reading image: {file_path}")
        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ImageValidationError(
                f"Failed to read image: {file_path}. "
                "File may be corrupted or in an unsupported format."
            )
        logger.debug(f"Image loaded: shape={img.shape}, dtype={img.dtype}")
        return img

    def read_image_bytes(self, data: bytes) -> np.ndarray:
        """
        Decode an image from raw bytes.

        Args:
            data: Raw image bytes (e.g., from an uploaded file).

        Returns:
            BGR image as a NumPy array.

        Raises:
            ImageValidationError: If decoding fails.
        """
        if not data or len(data) == 0:
            raise ImageValidationError("Empty image data received")

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageValidationError(
                "Failed to decode image from bytes. Data may be corrupted."
            )
        logger.debug(f"Image decoded from bytes: shape={img.shape}")
        return img

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate_image(self, img: np.ndarray) -> None:
        """
        Validate image dimensions, channels, and data type.

        Args:
            img: Image array to validate.

        Raises:
            ImageValidationError: If any validation check fails.
        """
        if img is None:
            raise ImageValidationError("Image is None")

        if img.ndim not in (2, 3):
            raise ImageValidationError(
                f"Invalid image dimensions: ndim={img.ndim} (expected 2 or 3)"
            )

        h, w = img.shape[:2]

        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            raise ImageValidationError(
                f"Image too small: {w}x{h} (minimum {MIN_IMAGE_DIM}x{MIN_IMAGE_DIM})"
            )

        if h > MAX_IMAGE_DIM or w > MAX_IMAGE_DIM:
            raise ImageValidationError(
                f"Image too large: {w}x{h} (maximum {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM})"
            )

        if img.ndim == 3:
            channels = img.shape[2]
            if channels not in SUPPORTED_CHANNELS:
                raise ImageValidationError(
                    f"Unsupported channel count: {channels} "
                    f"(supported: {SUPPORTED_CHANNELS})"
                )

        logger.debug(f"Image validated: {w}x{h}, dtype={img.dtype}")

    # ------------------------------------------------------------------
    # Convert color space
    # ------------------------------------------------------------------
    def convert_bgr_to_rgb(self, img: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to RGB color space.

        Args:
            img: BGR image array.

        Returns:
            RGB image array.
        """
        if img.ndim == 2:
            # Grayscale — convert to 3-channel RGB
            logger.debug("Converting grayscale to RGB")
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        channels = img.shape[2]
        if channels == 4:
            # BGRA to RGB (drop alpha)
            logger.debug("Converting BGRA to RGB")
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Standard BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to the target input dimensions.

        Uses letterboxing to preserve aspect ratio when dimensions differ
        significantly; otherwise performs a direct resize.

        Args:
            img: Input image array.

        Returns:
            Resized image array.
        """
        h, w = img.shape[:2]
        target_w, target_h = self.target_size

        if (w, h) == (target_w, target_h):
            logger.debug("Image already at target size, skipping resize")
            return img

        resized = cv2.resize(
            img,
            (target_w, target_h),
            interpolation=self.interpolation,
        )
        logger.debug(f"Resized image: {w}x{h} -> {target_w}x{target_h}")
        return resized

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values from [0, 255] to [0.0, 1.0].

        Args:
            img: Image array with uint8 values.

        Returns:
            Normalized image as float32 array.
        """
        normalized = img.astype(np.float32) / 255.0
        logger.debug(
            f"Normalized: min={normalized.min():.4f}, max={normalized.max():.4f}"
        )
        return normalized

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def preprocess(
        self,
        source,
        return_original: bool = False,
    ) -> dict:
        """
        Run the full preprocessing pipeline on an image.

        Args:
            source: File path (str/Path) or raw bytes.
            return_original: If True, include the original BGR image.

        Returns:
            Dictionary with:
                - 'image': Processed image array (RGB, resized, optionally normalized)
                - 'original_size': (width, height) of original image
                - 'target_size': (width, height) after resize
                - 'original' (optional): Original BGR image
        """
        # Step 1 — Read
        if isinstance(source, (str,)):
            img = self.read_image(source)
        elif isinstance(source, bytes):
            img = self.read_image_bytes(source)
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            from pathlib import Path as _P
            if isinstance(source, _P):
                img = self.read_image(str(source))
            else:
                raise ImageValidationError(
                    f"Unsupported source type: {type(source)}"
                )

        original = img.copy() if return_original else None
        original_h, original_w = img.shape[:2]

        # Step 2 — Validate
        self.validate_image(img)

        # Step 3 — BGR to RGB
        img = self.convert_bgr_to_rgb(img)

        # Step 4 — Resize
        img = self.resize_image(img)

        # Step 5 — Normalize
        if self.normalize:
            img = self.normalize_image(img)

        result = {
            "image": img,
            "original_size": (original_w, original_h),
            "target_size": self.target_size,
        }
        if return_original:
            result["original"] = original

        logger.info(
            f"Preprocessing complete: {original_w}x{original_h} -> "
            f"{self.target_size[0]}x{self.target_size[1]}"
        )
        return result
