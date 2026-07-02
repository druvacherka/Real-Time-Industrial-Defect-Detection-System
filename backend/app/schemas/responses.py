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
