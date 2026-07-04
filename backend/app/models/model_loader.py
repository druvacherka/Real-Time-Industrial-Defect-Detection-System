"""
YOLOv8 Model Loader Interface
===============================
Provides a clean abstraction over the YOLOv8 model for inference.
Designed so the real model can be swapped in later without changing the API.

Author: prajwaledu802-coder
Date: 2026-07-04
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

logger = logging.getLogger("defect_detection.model_loader")

# Default model configuration
DEFAULT_MODEL_PATH = Path("models/yolov8n_defects.pt")
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_INPUT_SIZE = (640, 640)

DEFECT_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}


class DetectionResult:
    """Single detection result from the model."""

    def __init__(
        self,
        class_name: str,
        class_id: int,
        confidence: float,
        bounding_box: List[int],
    ):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bounding_box = bounding_box

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bounding_box": self.bounding_box,
        }


class PredictionResponse:
    """Structured prediction response."""

    def __init__(
        self,
        detections: List[DetectionResult],
        processing_time_ms: float,
        image_size: tuple,
        model_name: str,
    ):
        self.detections = detections
        self.processing_time_ms = processing_time_ms
        self.image_size = image_size
        self.model_name = model_name

    def to_dict(self) -> dict:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "detection_count": len(self.detections),
            "processing_time": f"{self.processing_time_ms:.1f} ms",
            "image_size": list(self.image_size),
            "model": self.model_name,
        }


class YOLOv8ModelWrapper:
    """
    Wrapper for YOLOv8 model inference.

    Currently uses placeholder predictions. When a trained model is
    available, replace the `_run_inference` method to call the real
    YOLO model without modifying the API layer.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        self.model_name = "yolov8n_defects"

        logger.info(
            f"ModelWrapper initialized: path={model_path}, "
            f"conf={confidence_threshold}, iou={iou_threshold}"
        )

    def load_model(self) -> bool:
        """
        Load the YOLOv8 model from disk.

        Returns True if successful, False otherwise.
        When no trained model is available, initializes in placeholder mode.
        """
        logger.info(f"Loading model from {self.model_path}...")

        if self.model_path.exists():
            try:
                # Future: from ultralytics import YOLO
                # self.model = YOLO(str(self.model_path))
                logger.info("Model file found — placeholder mode active")
                self.is_loaded = True
            except Exception as exc:
                logger.error(f"Failed to load model: {exc}")
                return False
        else:
            logger.warning(
                f"Model file not found at {self.model_path}. "
                "Running in placeholder mode."
            )
            self.is_loaded = True  # Allow placeholder predictions

        return self.is_loaded

    def _run_inference(self, image: np.ndarray) -> List[DetectionResult]:
        """
        Run model inference on a preprocessed image.

        Currently returns placeholder detections.
        Replace this method body with real YOLO inference when ready.

        Args:
            image: Preprocessed image as NumPy array (RGB, resized).

        Returns:
            List of DetectionResult objects.
        """
        # ---- PLACEHOLDER PREDICTIONS ----
        # Simulates what the real model would return
        placeholder_detections = [
            DetectionResult(
                class_name="scratches",
                class_id=5,
                confidence=0.96,
                bounding_box=[120, 80, 240, 210],
            ),
            DetectionResult(
                class_name="inclusion",
                class_id=1,
                confidence=0.87,
                bounding_box=[50, 30, 180, 160],
            ),
        ]

        # Filter by confidence threshold
        filtered = [
            d for d in placeholder_detections
            if d.confidence >= self.confidence_threshold
        ]

        return filtered

    def predict(self, image: np.ndarray) -> PredictionResponse:
        """
        Run prediction on a preprocessed image.

        Args:
            image: Preprocessed NumPy array (RGB, float32 or uint8).

        Returns:
            PredictionResponse with detections and metadata.
        """
        if not self.is_loaded:
            self.load_model()

        start_time = time.time()

        # Get image dimensions
        if image.ndim == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        # Run inference
        detections = self._run_inference(image)

        processing_time = (time.time() - start_time) * 1000  # ms

        response = PredictionResponse(
            detections=detections,
            processing_time_ms=processing_time,
            image_size=(w, h),
            model_name=self.model_name,
        )

        logger.info(
            f"Prediction complete: {len(detections)} detections "
            f"in {processing_time:.1f} ms"
        )

        return response

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "is_loaded": self.is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "num_classes": len(DEFECT_CLASSES),
            "classes": DEFECT_CLASSES,
        }


# Module-level singleton for easy import
_model_instance: Optional[YOLOv8ModelWrapper] = None


def get_model() -> YOLOv8ModelWrapper:
    """Get or create the global model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = YOLOv8ModelWrapper()
        _model_instance.load_model()
    return _model_instance
