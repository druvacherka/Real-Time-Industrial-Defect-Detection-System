"""
schemas/responses.py — Pydantic Response Models
=================================================
Defines the response shapes for all API endpoints.
These models are used by FastAPI for automatic JSON serialisation
and Swagger documentation.
"""

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Response schema for the root endpoint (GET /)."""

    message: str = Field(
        ...,
        description="Welcome message from the API",
        json_schema_extra={"example": "Industrial Defect Detection API"},
    )


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint (GET /health)."""

    status: str = Field(
        ...,
        description="Current service status",
        json_schema_extra={"example": "healthy"},
    )
    service: str = Field(
        ...,
        description="Name of the service",
        json_schema_extra={"example": "Industrial Defect Detection"},
    )
    version: str = Field(
        ...,
        description="API version string",
        json_schema_extra={"example": "1.0.0"},
    )


class DetectionResult(BaseModel):
    """Schema for an individual defect bounding box detection."""

    class_name: str = Field(..., alias="class", description="Name of the detected defect class")
    confidence: float = Field(..., description="Model confidence score")
    bounding_box: list[int] = Field(
        ...,
        description="Bounding box coordinates [xmin, ymin, xmax, ymax] in pixels",
        json_schema_extra={"example": [120, 80, 260, 210]},
    )

    class Config:
        populate_by_name = True


class PredictionDetails(BaseModel):
    """Detailed prediction schema including list of detections and timing."""

    detections: list[DetectionResult] = Field(..., description="List of detected defect regions")
    processing_time: str = Field(..., description="Processing and inference duration")


class UploadImageResponse(BaseModel):
    """Response schema for the image upload and prediction endpoint."""

    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(..., description="Status of the upload/processing")
    message: str = Field(..., description="Detailed status message")
    prediction: PredictionDetails | None = Field(default=None, description="Detection prediction details")


