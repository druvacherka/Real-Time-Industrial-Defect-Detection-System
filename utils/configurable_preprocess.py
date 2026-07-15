"""
Configurable Preprocessing Service
=================================
Loads configuration parameters from configs/preprocessing.yaml and applies
optimized image transformations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import yaml
import numpy as np

from utils.logger import get_logger

logger = get_logger("configurable_preprocess")

class ConfigurablePreprocessor:
    """
    Service that processes images dynamically based on configuration yaml parameters.
    """
    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA
    }
    
    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "preprocessing.yaml"
            
        self.config = self.load_config(config_path)
        self.width = self.config.get("resize", {}).get("width", 200)
        self.height = self.config.get("resize", {}).get("height", 200)
        
        interp_name = self.config.get("resize", {}).get("interpolation", "bilinear")
        self.interpolation = self.INTERPOLATION_MAP.get(interp_name, cv2.INTER_LINEAR)
        
        norm_conf = self.config.get("normalization", {})
        self.norm_enabled = norm_conf.get("enabled", True)
        self.norm_type = norm_conf.get("type", "standard")
        self.mean = norm_conf.get("mean", [0.485, 0.456, 0.406])
        self.std = norm_conf.get("std", [0.229, 0.224, 0.225])
        self.output_format = self.config.get("output", {}).get("format", ".png")
        
    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        """Loads yaml configurations from the file path."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Could not load config path: %s. Using defaults. Error: %s", path, exc)
            return {}
            
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize and standardize/normalize a single image frame/numpy array.
        """
        # Resize
        h, w = img.shape[:2]
        if (w, h) != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height), interpolation=self.interpolation)
            
        # Normalize
        if self.norm_enabled:
            # Convert BGR to RGB for standard z-scoring (if 3-channels)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            img = img.astype(np.float32) / 255.0
            
            if self.norm_type == "standard":
                mean_arr = np.array(self.mean, dtype=np.float32)
                std_arr = np.array(self.std, dtype=np.float32)
                # Thread-safe operations
                img = (img - mean_arr) / std_arr
                
            # Replace non-finite items (NaN/Inf)
            if not np.isfinite(img).all():
                np.nan_to_num(img, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
                
        return img
