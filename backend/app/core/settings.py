"""
Application Settings using Pydantic BaseSettings
==================================================
Centralized configuration management with .env support.

Author: prajwaledu802-coder
Date: 2026-07-05
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.

    Configuration hierarchy (highest to lowest priority):
        1. Environment variables
        2. .env file
        3. Default values defined here
    """

    # ---- API Configuration ----
    API_TITLE: str = Field(
        default="Real-Time Industrial Defect Detection API",
        description="Title displayed in Swagger docs",
    )
    API_VERSION: str = Field(
        default="0.2.0",
        description="Current API version",
    )
    API_DESCRIPTION: str = Field(
        default="FastAPI backend for YOLOv8-based industrial surface defect detection",
    )
    API_HOST: str = Field(default="0.0.0.0", description="Server bind address")
    API_PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    # ---- Model Configuration ----
    MODEL_PATH: str = Field(
        default="models/yolov8n_defects.pt",
        description="Path to the YOLOv8 model weights file",
    )
    MODEL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for detections",
    )
    MODEL_IOU_THRESHOLD: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS",
    )
    MODEL_DEVICE: str = Field(
        default="auto",
        description="Inference device: 'cpu', 'cuda', 'auto'",
    )

    # ---- Upload Configuration ----
    UPLOAD_DIR: str = Field(
        default="backend/uploads",
        description="Directory for temporarily stored uploaded images",
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=10,
        description="Maximum allowed upload file size in MB",
    )

    # ---- Image Preprocessing ----
    INPUT_IMAGE_WIDTH: int = Field(default=640, description="YOLO input width")
    INPUT_IMAGE_HEIGHT: int = Field(default=640, description="YOLO input height")
    NORMALIZE_IMAGES: bool = Field(default=True, description="Normalize pixel values")

    # ---- Logging Configuration ----
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        description="Python logging format string",
    )
    LOG_DIR: str = Field(
        default="logs",
        description="Directory for log files",
    )

    # ---- CORS ----
    CORS_ORIGINS: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
