"""
Model Service Layer
====================
Provides a high-level service interface for managing the YOLOv8 model.
Coordinates loading, inference on different media types, and memory unloading.

Author: prajwaledu802-coder
Date: 2026-07-07
"""

import time
import logging
import gc
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from app.models.model_loader import get_model, YOLOv8ModelWrapper, PredictionResponse

logger = logging.getLogger("defect_detection.model_service")


class ModelService:
    """
    Service layer providing unified access to YOLOv8 inference operations.
    Handles model lifecycle management (loading/unloading) and prediction routing.
    """

    def __init__(self, model_path: Optional[Path] = None, conf_threshold: float = 0.25):
        import threading
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model_wrapper: Optional[YOLOv8ModelWrapper] = None
        self._lock = threading.Lock()

    def load_model(self) -> bool:
        """
        Load the YOLOv8 model wrapper into memory. Thread-safe implementation.
        """
        if self.model_wrapper is not None and getattr(self.model_wrapper, "is_loaded", False):
            return True
            
        with self._lock:
            # Double checked locking pattern
            if self.model_wrapper is not None and getattr(self.model_wrapper, "is_loaded", False):
                return True
            try:
                logger.info("ModelService: lazy loading YOLO model wrapper...")
                if self.model_path:
                    self.model_wrapper = YOLOv8ModelWrapper(
                        model_path=self.model_path,
                        confidence_threshold=self.conf_threshold,
                    )
                    self.model_wrapper.load_model()
                else:
                    self.model_wrapper = get_model()
                logger.info("ModelService: model loaded successfully")
                return True
            except Exception as exc:
                logger.error(f"ModelService: failed to load model: {exc}", exc_info=True)
                self.model_wrapper = None
                return False

    def predict_image(self, processed_image: np.ndarray, conf_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Run inference on preprocessed image array with optional confidence threshold override.
        """
        if self.model_wrapper is None:
            self.load_model()
            
        if not self.model_wrapper:
            logger.error("ModelService: predict_image failed (model not loaded)")
            raise RuntimeError("YOLO model is not loaded in memory")

        original_conf = self.model_wrapper.confidence_threshold
        if conf_threshold is not None:
            self.model_wrapper.confidence_threshold = conf_threshold

        try:
            start_time = time.time()
            prediction: PredictionResponse = self.model_wrapper.predict(processed_image)
            duration_ms = (time.time() - start_time) * 1000

            result = prediction.to_dict()
            result["status"] = "success"
            result["inference_service_time"] = f"{duration_ms:.2f} ms"
            return result
        except Exception as exc:
            logger.error(f"ModelService: image prediction failed: {exc}")
            raise RuntimeError(f"Inference engine failure: {str(exc)}")
        finally:
            self.model_wrapper.confidence_threshold = original_conf

    def predict_video(
        self, 
        video_path: Path, 
        conf_threshold: Optional[float] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run inference on video file frame-by-frame using OpenCV with optional confidence threshold override.
        Accumulates detection results across frames and optionally writes annotated video to output_path.
        """
        if self.model_wrapper is None:
            self.load_model()
            
        if not self.model_wrapper:
            logger.error("ModelService: predict_video failed (model not loaded)")
            raise RuntimeError("YOLO model is not loaded in memory")

        original_conf = self.model_wrapper.confidence_threshold
        if conf_threshold is not None:
            self.model_wrapper.confidence_threshold = conf_threshold

        start_time = time.time()
        logger.info(f"ModelService: starting video inference pipeline for {video_path}")
        
        # Color mapping (BGR format) corresponding to Javascript class colors:
        CLASS_COLORS_BGR = {
            "crazing": (68, 68, 239),       # Red
            "inclusion": (11, 158, 245),    # Orange
            "patches": (129, 185, 16),      # Green
            "pitted_surface": (212, 182, 6), # Cyan
            "rolled-in_scale": (241, 102, 99), # Indigo
            "scratches": (246, 92, 139)     # Purple
        }

        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("Could not open video file via OpenCV")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0.0

            logger.info(
                f"ModelService: loaded video metadata: resolution={width}x{height}, "
                f"fps={fps:.2f}, total_frames={frame_count}, duration={duration:.2f}s"
            )
            
            # Setup VideoWriter if output_path is provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
            
            total_detections = 0
            class_counts = {}
            frame_idx = 0
            processed_frames = 0
            sample_rate = 5
            
            active_detections = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
                    predictions = self.model_wrapper.predict_batch([frame_resized])
                    active_detections = predictions[0].detections if predictions else []
                    
                    for det in active_detections:
                        total_detections += 1
                        class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                    processed_frames += 1

                if out:
                    if active_detections:
                        for det in active_detections:
                            box = det.bounding_box
                            x_min = int(box[0] * (width / 640.0))
                            y_min = int(box[1] * (height / 640.0))
                            x_max = int(box[2] * (width / 640.0))
                            y_max = int(box[3] * (height / 640.0))
                            
                            color = CLASS_COLORS_BGR.get(det.class_name, (99, 102, 241))
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                            
                            label = f"{det.class_name} {det.confidence:.2f}"
                            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    out.write(frame)

                frame_idx += 1

            cap.release()
            if out:
                out.release()
            logger.info("ModelService: released OpenCV resources")

            # Format detection summary
            from app.models.model_loader import DEFECT_CLASSES
            name_to_id = {v: k for k, v in DEFECT_CLASSES.items()}
            detections_summary = [
                {
                    "class": name,
                    "class_id": name_to_id.get(name, -1),
                    "count": count
                }
                for name, count in class_counts.items()
            ]

            processing_time = time.time() - start_time
            logger.info(
                f"ModelService: video inference completed: processed {processed_frames}/{frame_count} frames, "
                f"detected {total_detections} total defects in {processing_time:.2f} s"
            )

            return {
                "detection_summary": detections_summary,
                "total_detections": total_detections,
                "processing_time": f"{processing_time:.2f} s",
                "frame_count": frame_count,
                "fps": round(fps, 2) if fps > 0 else 30.0,
                "video_duration_seconds": round(duration, 2),
                "video_resolution": [width, height],
                "model": self.model_wrapper.model_name,
                "status": "success",
            }
        except Exception as exc:
            logger.error(f"ModelService: video prediction failed: {exc}")
            raise RuntimeError(f"Video processing/inference engine failure: {str(exc)}")
        finally:
            self.model_wrapper.confidence_threshold = original_conf

    def unload_model(self) -> None:
        """
        Unload the YOLO model from memory and trigger garbage collection.
        """
        logger.info("ModelService: initiating model unloading sequence...")
        if self.model_wrapper:
            self.model_wrapper.model = None
            self.model_wrapper.is_loaded = False
        self.model_wrapper = None
        gc.collect()
        logger.info("ModelService: memory freed successfully post-unload")


# Global model service singleton
_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Retrieve or initialize the global model service with tracking logs."""
    global _model_service_instance
    if _model_service_instance is None:
        logger.info("ModelService: Instantiating global singleton instance")
        _model_service_instance = ModelService()
    return _model_service_instance
