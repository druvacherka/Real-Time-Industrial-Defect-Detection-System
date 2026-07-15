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

from app.core.settings import get_settings

from app.core.inference_config import inference_config

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
MODEL_CONFIDENCE_THRESHOLD = inference_config.conf_threshold
MODEL_IOU_THRESHOLD = inference_config.iou_threshold
MODEL_DEVICE = settings.MODEL_DEVICE

# ── Image Preprocessing ───────────────────────────────────────────────────
INPUT_SIZE = inference_config.image_size
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


def validate_production_config(logger=None):
    """
    Validate environment variables and folder setups for production.
    Raises warnings or errors for unsafe configurations.
    """
    if logger is None:
        logger = logging.getLogger("defect_detection.config_validation")
        
    logger.info("Starting production configuration validation...")

    # 1. API Key Security Check
    if settings.API_KEY == "industrial-defect-secret-key":
        logger.warning(
            "SECURITY WARNING: The API_KEY is set to the default fallback value. "
            "Please configure a strong custom API_KEY in production environment settings."
        )
    else:
        logger.info("API Key validation: Custom API key is set.")

    # 2. Check directory permissions
    for directory, name in [(LOG_DIR, "Log"), (UPLOAD_DIR, "Upload")]:
        if directory.exists():
            if not os.access(directory, os.W_OK):
                logger.error(f"CRITICAL: {name} directory '{directory}' is not writable!")
            else:
                logger.info(f"{name} directory '{directory}' is verified writable.")
        else:
            logger.warning(f"{name} directory '{directory}' does not exist yet.")

    # 3. Model Weights Check
    weights = Path(MODEL_WEIGHTS_PATH)
    if not weights.exists():
        logger.warning(
            f"MODEL PATH WARNING: Bounding box model weights not found at '{MODEL_WEIGHTS_PATH}'. "
            "System will fallback to running in mock/placeholder prediction mode."
        )
    else:
        logger.info(f"Model weights verified at '{MODEL_WEIGHTS_PATH}'. Size: {weights.stat().st_size / (1024*1024):.2f} MB")
