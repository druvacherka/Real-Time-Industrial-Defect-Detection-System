"""
API Response Schemas
=====================
Pydantic models for structured API responses.

Optimized on: 2026-07-12 by prajwaledu802-coder
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="0.1.0")
    service: str = Field(default="Industrial Defect Detection", example="Industrial Defect Detection")
    cpu_percent: float = Field(default=0.0, example=12.5)
    memory_percent: float = Field(default=0.0, example=45.2)
    queue_size: Optional[int] = Field(default=None, example=2)
    jobs_processed: Optional[int] = Field(default=None, example=120)
    avg_queue_wait_ms: Optional[float] = Field(default=None, example=15.4)
    uptime_seconds: Optional[float] = Field(default=None, example=3600.0)
    total_requests_processed: Optional[int] = Field(default=None, example=150)
    analytics: Optional[dict] = Field(default=None, description="Detailed backend analytics and metrics")


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


class FrameDetectionSummary(BaseModel):
    """Detection summary for a specific defect class in a video."""
    class_name: str = Field(..., alias="class", example="scratches")
    class_id: int = Field(..., example=5)
    count: int = Field(..., example=3)

    class Config:
        populate_by_name = True


class VideoPredictionDetails(BaseModel):
    """Full prediction result for video."""
    detection_summary: List[FrameDetectionSummary] = Field(default_factory=list)
    total_detections: int = Field(default=0, example=15)
    processing_time: str = Field(default="0 ms", example="2.3s")
    frame_count: int = Field(default=0, example=120)
    fps: float = Field(default=0.0, example=30.0)
    video_duration_seconds: float = Field(default=0.0, example=4.0)
    video_resolution: List[int] = Field(default_factory=lambda: [1920, 1080])
    model: str = Field(default="yolov8n_defects")
    annotated_video_url: Optional[str] = Field(default=None, example="/predict/video/download/annotated_some_video.mp4")


class UploadVideoResponse(BaseModel):
    """Response for the video upload prediction endpoint."""
    request_id: str = Field(..., example="b2c3d4e5f6g7")
    filename: str = Field(..., example="sample_video.mp4")
    status: str = Field(..., example="success")
    message: str = Field(..., example="Video processed successfully")
    file_size_mb: float = Field(default=0.0, example=4.52)
    processing_time_ms: float = Field(default=0.0, example=1250.5)
    prediction: Optional[VideoPredictionDetails] = None


class LiveStreamPredictionDetails(BaseModel):
    """Metadata response for active live stream defect detection."""
    status: str = Field(..., example="connected")
    source: str = Field(..., example="rtsp://127.0.0.1:8554/live")
    fps: float = Field(default=0.0, example=30.0)
    video_resolution: List[int] = Field(default_factory=lambda: [1280, 720])
    model: str = Field(default="yolov8n_defects")


class UploadLiveResponse(BaseModel):
    """Response for the live prediction endpoint."""
    request_id: str = Field(..., example="c3d4e5f6g7h8")
    status: str = Field(..., example="success")
    message: str = Field(..., example="Live stream connection established successfully")
    prediction: Optional[LiveStreamPredictionDetails] = None
