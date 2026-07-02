"""
config.py — Application Configuration
=======================================
Centralised configuration loader for the defect detection backend.
Reads from environment variables with sensible defaults.
"""

import os
from pathlib import Path


# ── Path Constants ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
APP_DIR = BASE_DIR / "app"
LOG_DIR = BASE_DIR / "logs"

# Make sure the logs directory exists on startup
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Application Metadata ───────────────────────────────────────────────────
APP_NAME = "Industrial Defect Detection API"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "REST API for real-time surface defect detection on metal products "
    "using YOLOv8 object detection."
)

# ── Server Settings ────────────────────────────────────────────────────────
HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", 8000))
DEBUG = os.getenv("APP_DEBUG", "false").lower() == "true"

# ── Model Settings ─────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = os.getenv(
    "MODEL_WEIGHTS",
    str(BASE_DIR.parent / "weights" / "best.pt"),
)
MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONF_THRESHOLD", 0.5))
MODEL_IOU_THRESHOLD = float(os.getenv("MODEL_IOU_THRESHOLD", 0.45))

# ── CORS Settings ──────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Upload Limits ──────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 10))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}
