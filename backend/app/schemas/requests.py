"""
API Request Schemas
====================
Pydantic models for incoming API requests.

Author: prajwaledu802-coder
Date: 2026-07-07
"""

from pydantic import BaseModel, Field


class LiveStreamRequest(BaseModel):
    """Request model for initiating live stream defect detection."""
    source: str = Field(
        ...,
        example="rtsp://127.0.0.1:8554/live",
        description="RTSP, RTMP, HTTP live stream URL, or webcam device index",
    )
    conf_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        example=0.25,
        description="Confidence threshold for detections",
    )
