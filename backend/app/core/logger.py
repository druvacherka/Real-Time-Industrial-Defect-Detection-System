"""
logger.py — Centralized Logging Configuration
================================================
Sets up structured logging for the entire backend application.
Logs are written to both the console and a rotating log file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import LOG_DIR, APP_NAME, DEBUG


# ── Log file path ──────────────────────────────────────────────────────────
LOG_FILE = LOG_DIR / "app.log"

# ── Log format ─────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | "
    "%(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> logging.Logger:
    """
    Configure and return the root application logger.

    Creates two handlers:
      - Console handler (stdout) for development visibility
      - Rotating file handler (logs/app.log) for persistent logging

    Returns:
        Configured Logger instance.
    """
    # Determine log level based on debug mode
    log_level = logging.DEBUG if DEBUG else logging.INFO

    # Create the logger
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console Handler ─────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler (rotating, max 5 MB, keep 3 backups) ──────────────
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging initialised — level=%s, file=%s", log_level, LOG_FILE)
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a child logger scoped to a specific module.

    Args:
        name: Optional module name for the child logger.

    Returns:
        Logger instance.
    """
    base = APP_NAME
    if name:
        return logging.getLogger(f"{base}.{name}")
    return logging.getLogger(base)
