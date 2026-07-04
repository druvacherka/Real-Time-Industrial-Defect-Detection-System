"""
API Response Schemas
=====================
Pydantic models for structured API responses.

Author: prajwaledu802-coder
Date: 2026-07-04
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="0.1.0")


class RootResponse(BaseModel):
    """Root endpoint response."""
    project: str
    description: str
    docs_url: str


class DetectionItem(BaseModel):
    """Single defect detection result."""
    class_name: str = Field(..., alias="class", example="scratches")
    class_id: int = Field(..., example=5)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.96)
    bounding_box: List[int] = Field(
        ...,
        min_items=4,
        max_items=4,
        example=[120, 80, 240, 210],
        description="Bounding box coordinates [x1, y1, x2, y2]",
    )

    class Config:
        populate_by_name = True


class PredictionDetails(BaseModel):
    """Full prediction result with detections and metadata."""
    detections: List[DetectionItem] = Field(default_factory=list)
    detection_count: int = Field(default=0, example=2)
    processing_time: str = Field(default="0 ms", example="34 ms")
    image_size: List[int] = Field(default_factory=lambda: [640, 640])
    model: str = Field(default="yolov8n_defects")


class UploadImageResponse(BaseModel):
    """Response for the image upload prediction endpoint."""
    request_id: str = Field(..., example="a1b2c3d4e5f6")
    filename: str = Field(..., example="sample.jpg")
    status: str = Field(..., example="success")
    message: str = Field(..., example="Image uploaded successfully")
    file_size_mb: float = Field(default=0.0, example=1.234)
    processing_time_ms: float = Field(default=0.0, example=34.5)
    prediction: Optional[PredictionDetails] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., example="unsupported_format")
    message: str = Field(..., example="File format '.gif' is not supported.")
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    model_name: str = Field(..., example="yolov8n_defects")
    model_path: str
    is_loaded: bool
    confidence_threshold: float
    iou_threshold: float
    device: str
    num_classes: int
    classes: dict
