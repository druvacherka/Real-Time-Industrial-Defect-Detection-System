"""
settings.py — Pydantic Settings Management
=============================================
Typed application settings using Pydantic's BaseSettings.
Reads from environment variables and .env files automatically.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """
    Application-wide settings with environment variable support.

    Pydantic will automatically read matching environment variables
    (case-insensitive) or fall back to the defaults specified here.
    """

    # ── App Metadata ────────────────────────────────────────────────────
    app_name: str = Field(
        default="Industrial Defect Detection API",
        description="Display name for the API service",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Current API version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, auto-reload)",
    )

    # ── Server ──────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8000, description="Server port")

    # ── Model Inference ─────────────────────────────────────────────────
    model_weights_path: str = Field(
        default="weights/best.pt",
        description="Path to the YOLOv8 trained weights file",
    )
    model_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections",
    )
    model_iou_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for non-max suppression",
    )

    # ── CORS ────────────────────────────────────────────────────────────
    allowed_origins: list[str] = Field(
        default=["*"],
        description="List of allowed CORS origins",
    )

    # ── Upload ──────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=10,
        description="Maximum upload file size in megabytes",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance — import this wherever settings are needed
settings = AppSettings()
