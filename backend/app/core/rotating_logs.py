"""
Rotating logs configurator to manage file rotation policies.
Author: prajwaledu802-coder

Optimized for Prometheus monitoring: 2026-07-12
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def configure_rotating_logger(name: str, log_file: Path, max_bytes: int = 5*1024*1024, backup_count: int = 3):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
