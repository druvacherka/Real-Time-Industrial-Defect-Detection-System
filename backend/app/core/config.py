"""
config.py — Application Configuration
=======================================
Centralized configuration loader for the defect detection backend.
Now powered by Pydantic BaseSettings with .env file support.

Author: prajwaledu802-coder
Date: 2026-07-05
"""

import os
import logging
from pathlib import Path

from backend.app.core.settings import get_settings

# Load settings singleton
settings = get_settings()

# ── Path Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
APP_DIR = BASE_DIR / "app"
LOG_DIR = Path(settings.LOG_DIR)
UPLOAD_DIR = Path(settings.UPLOAD_DIR)

# Create required directories on import
LOG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Application Metadata ───────────────────────────────────────────────────
APP_NAME = settings.API_TITLE
APP_VERSION = settings.API_VERSION
APP_DESCRIPTION = settings.API_DESCRIPTION

# ── Server Settings ────────────────────────────────────────────────────────
HOST = settings.API_HOST
PORT = settings.API_PORT
DEBUG = settings.DEBUG

# ── Model Settings ─────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = settings.MODEL_PATH
MODEL_CONFIDENCE_THRESHOLD = settings.MODEL_CONFIDENCE_THRESHOLD
MODEL_IOU_THRESHOLD = settings.MODEL_IOU_THRESHOLD
MODEL_DEVICE = settings.MODEL_DEVICE

# ── Image Preprocessing ───────────────────────────────────────────────────
INPUT_SIZE = (settings.INPUT_IMAGE_WIDTH, settings.INPUT_IMAGE_HEIGHT)
NORMALIZE_IMAGES = settings.NORMALIZE_IMAGES

# ── CORS Settings ──────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [o.strip() for o in settings.CORS_ORIGINS.split(",")]

# ── Upload Limits ──────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = settings.MAX_UPLOAD_SIZE_MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}


def configure_logging():
    """Configure application-wide logging based on settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "app.log", mode="a"),
        ],
    )
    logger = logging.getLogger("defect_detection")
    logger.setLevel(log_level)
    logger.info(
        f"Logging configured: level={settings.LOG_LEVEL}, dir={LOG_DIR}"
    )
    return logger
