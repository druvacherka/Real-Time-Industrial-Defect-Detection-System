"""
YOLO Inference Service
======================
Handles loading the YOLO model and running prediction inference.
Abstractions are designed to allow seamless transition to a fully trained model.

Author: prajwaledu802-coder
Date: 2026-07-05
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

# Import the model loader from the app models directory
from backend.app.models.model_loader import get_model, YOLOv8ModelWrapper, PredictionResponse

logger = logging.getLogger("defect_detection.inference")


class InferenceService:
    """
    Service layer coordinating model load and prediction executions.

    Integrates with YOLOv8ModelWrapper and configures custom confidence thresholds,
    tracking performance metrics (e.g. inference time).
    """

    def __init__(self, model_path: Optional[Path] = None, conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model_wrapper: Optional[YOLOv8ModelWrapper] = None
        self._initialize_model()

    def _initialize_model(self):
        """Load and configure the underlying YOLO model wrapper."""
        try:
            logger.info("Initializing inference service model wrapper...")
            # Use module singleton or create a configured wrapper instance
            if self.model_path:
                self.model_wrapper = YOLOv8ModelWrapper(
                    model_path=self.model_path,
                    confidence_threshold=self.conf_threshold,
                )
                self.model_wrapper.load_model()
            else:
                self.model_wrapper = get_model()
            logger.info("YOLO model successfully configured for inference service")
        except Exception as exc:
            logger.error(f"Graceful initialization failed: {exc}")
            self.model_wrapper = None

    def predict_image(self, processed_image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a preprocessed image array.

        Args:
            processed_image: Resized and normalized RGB NumPy image array.

        Returns:
            Dictionary containing prediction detections, metadata, and timing.
        """
        if self.model_wrapper is None:
            logger.error("Prediction attempted, but model wrapper is not initialized")
            return {
                "detections": [],
                "detection_count": 0,
                "processing_time": "0 ms",
                "status": "error",
                "error_message": "Inference model not loaded",
            }

        start_time = time.time()
        try:
            logger.info("Executing YOLO inference service predict_image...")
            # Predict expects preprocessed image
            prediction: PredictionResponse = self.model_wrapper.predict(processed_image)
            inference_time_ms = (time.time() - start_time) * 1000

            result = prediction.to_dict()
            result["status"] = "success"
            # Overlay custom inference-service tracking time
            result["inference_service_time"] = f"{inference_time_ms:.2f} ms"
            logger.info(
                f"Inference execution successful: detections={result['detection_count']} "
                f"time={inference_time_ms:.2f}ms"
            )
            return result
        except Exception as exc:
            logger.error(f"Inference prediction failed: {exc}")
            return {
                "detections": [],
                "detection_count": 0,
                "processing_time": "0 ms",
                "status": "error",
                "error_message": f"Inference execution failed: {str(exc)}",
            }


# Module level helper for easy dependency injection
_inference_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Retrieve or initialize the global inference service."""
    global _inference_service_instance
    if _inference_service_instance is None:
        _inference_service_instance = InferenceService()
    return _inference_service_instance
