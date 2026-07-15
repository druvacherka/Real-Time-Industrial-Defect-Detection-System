"""
Inference Configuration Module
=============================
Author: prajwaledu802-coder
Date: 2026-07-15
"""

import os
import logging
from typing import Tuple

logger = logging.getLogger("defect_detection.inference_config")

class InferenceConfig:
    """
    Manages custom dynamic inference configuration settings,
    loading values from environment variables with safety validations.
    """
    def __init__(self) -> None:
        # Confidence threshold [0.0, 1.0]
        self.conf_threshold: float = self._load_float("INFERENCE_CONFIDENCE_THRESHOLD", 0.25, 0.0, 1.0)
        # IoU threshold [0.0, 1.0]
        self.iou_threshold: float = self._load_float("INFERENCE_IOU_THRESHOLD", 0.45, 0.0, 1.0)
        # Image dimensions (width, height)
        self.image_width: int = self._load_int("INFERENCE_IMAGE_WIDTH", 640, 32, 2048)
        self.image_height: int = self._load_int("INFERENCE_IMAGE_HEIGHT", 640, 32, 2048)
        
        logger.info(
            "InferenceConfig initialized: conf_threshold=%.2f, iou_threshold=%.2f, img_size=(%d, %d)",
            self.conf_threshold,
            self.iou_threshold,
            self.image_width,
            self.image_height
        )

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.image_width, self.image_height)

    def _load_float(self, env_var: str, default: float, min_val: float, max_val: float) -> float:
        val_str = os.getenv(env_var)
        if val_str is None:
            return default
        try:
            val = float(val_str)
            if not (min_val <= val <= max_val):
                logger.warning("Env %s=%s out of bounds [%.2f, %.2f]. Using default %s.", env_var, val_str, min_val, max_val, default)
                return default
            return val
        except ValueError:
            logger.warning("Env %s=%s not a valid float. Using default %s.", env_var, val_str, default)
            return default

    def _load_int(self, env_var: str, default: int, min_val: int, max_val: int) -> int:
        val_str = os.getenv(env_var)
        if val_str is None:
            return default
        try:
            val = int(val_str)
            if not (min_val <= val <= max_val):
                logger.warning("Env %s=%s out of bounds [%d, %d]. Using default %d.", env_var, val_str, min_val, max_val, default)
                return default
            return val
        except ValueError:
            logger.warning("Env %s=%s not a valid int. Using default %d.", env_var, val_str, default)
            return default

# Global instance
inference_config = InferenceConfig()
