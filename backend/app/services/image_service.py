"""
services/image_service.py — Image Preprocessing Service
=========================================================
Handles loading, validation, and preprocessing of uploaded images
before passing them to the YOLOv8 model for inference.
"""

from pathlib import Path
import cv2
import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessingService:
    """Service to handle CV-based image validation and resizing operations."""

    def __init__(self, target_size: tuple[int, int] = (640, 640)):
        """
        Initialize the image service.

        Args:
            target_size: The (width, height) expected by the YOLOv8 model.
        """
        self.target_size = target_size

    def preprocess_image(self, file_path: Path) -> np.ndarray:
        """
        Read, validate, and preprocess an image for model prediction.

        Tasks:
          1. Read the image using OpenCV.
          2. Validate image dimensions and integrity.
          3. Convert color space from BGR (OpenCV default) to RGB.
          4. Resize to target YOLO input dimensions (640x640).

        Args:
            file_path: Path to the locally saved image file.

        Returns:
            Preprocessed NumPy array of shape (height, width, channels) in RGB.

        Raises:
            ValueError: If the image is corrupted, empty, or unreadable.
        """
        logger.info("Starting preprocessing for image: %s", file_path.name)

        # ── 1. Read the image using OpenCV ──────────────────────────────────
        if not file_path.exists():
            logger.error("Image file not found: %s", file_path)
            raise ValueError(f"Image file not found: {file_path}")

        image = cv2.imread(str(file_path))

        # ── 2. Validate image integrity ─────────────────────────────────────
        if image is None or image.size == 0:
            logger.error("Failed to read image or image is empty: %s", file_path.name)
            raise ValueError("Corrupted, empty, or unreadable image file.")

        h, w, c = image.shape
        logger.debug(
            "Original image loaded successfully — dimensions: %dx%d with %d channels",
            w, h, c
        )

        # ── 3. Convert BGR to RGB ───────────────────────────────────────────
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.error("Failed to convert image color space: %s", str(exc))
            raise ValueError("Failed to process image color channels.") from exc

        # ── 4. Resize to target size (e.g. 640x640) ──────────────────────────
        try:
            resized_image = cv2.resize(
                image_rgb,
                self.target_size,
                interpolation=cv2.INTER_LINEAR
            )
            logger.info(
                "Image preprocessing completed successfully. Resized to: %dx%d",
                self.target_size[0],
                self.target_size[1]
            )
            return resized_image
        except Exception as exc:
            logger.error("Failed to resize image: %s", str(exc))
            raise ValueError("Failed to scale image to model input size.") from exc


# Singleton instance
image_preprocessor = ImagePreprocessingService()
