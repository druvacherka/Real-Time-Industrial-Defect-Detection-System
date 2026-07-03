"""
models/model_loader.py — AI Model Loader and Prediction Interface
===================================================================
Handles model loading and runs inference using placeholder logic.
Designed to be easily swapped with the actual YOLOv8 model in the future.
"""

import time
import numpy as np

from app.core.logger import get_logger
from app.core.config import MODEL_WEIGHTS_PATH

logger = get_logger(__name__)


class YOLOv8ModelWrapper:
    """Wrapper class for loading and running predictions on the YOLOv8 model."""

    def __init__(self):
        self.model = None
        self.classes = [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled_in_scale",
            "scratches"
        ]

    def load_model(self) -> None:
        """
        Load the YOLOv8 model weights.
        Currently uses placeholder loading logic.
        """
        logger.info("Initializing model load from path: %s", MODEL_WEIGHTS_PATH)
        # Placeholder delay simulating model loading from disk
        time.sleep(0.5)
        logger.info("YOLOv8 Model loaded successfully (Mock mode).")

    def predict(self, preprocessed_image: np.ndarray) -> list[dict]:
        """
        Run inference on the preprocessed image.
        Currently uses placeholder prediction logic returning static mock defects.

        Args:
            preprocessed_image: Resized RGB image as NumPy array.

        Returns:
            List of detected defect dictionaries.
        """
        logger.info("Running defect detection inference on image matrix of shape: %s", preprocessed_image.shape)

        # Simulating processing latency
        time.sleep(0.035)

        # Mock prediction returning a defect for demonstration
        mock_detections = [
            {
                "class_name": "scratch",
                "confidence": 0.95,
                "bounding_box": [120, 80, 260, 210]
            }
        ]

        logger.info("Inference completed — detected %d defect(s)", len(mock_detections))
        return mock_detections


# Singleton instance
model_wrapper = YOLOv8ModelWrapper()
