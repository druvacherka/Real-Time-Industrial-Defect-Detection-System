"""
Dataset Quality Scoring Module
=============================
Computes metrics including annotation coverage, resolution distributions,
class distribution entropy, and a composite quality score.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("quality_scoring")

def calculate_shannon_entropy(counts: List[int]) -> float:
    """
    Calculate Shannon entropy to measure class imbalance.
    Higher entropy indicates a more balanced dataset.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    probabilities = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log2(p) for p in probabilities))

def compute_quality_score(
    resolution_score: float,
    blur_score: float,
    balance_score: float,
    annotation_error_rate: float
) -> float:
    """
    Compute a composite dataset quality score in [0, 100].
    """
    # Resolution score (max 25) + blur score (max 25) + balance score (max 30) - error penalty (max 20)
    score = (resolution_score * 25.0) + (blur_score * 25.0) + (balance_score * 30.0)
    penalty = annotation_error_rate * 100.0 * 2.0  # double the error percentage as penalty
    final_score = max(0.0, min(100.0, score - penalty))
    return float(final_score)

def parse_box_coverage(
    label_path: Path
) -> List[float]:
    """
    Calculate the ratio of bounding box area relative to total image area.
    YOLO boxes are (cx, cy, w, h) normalized coordinates, so area = w * h.
    """
    ratios = []
    if not label_path.exists():
        return ratios
    try:
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            return ratios
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                w = float(parts[3])
                h = float(parts[4])
                if 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                    ratios.append(w * h)
    except Exception as exc:
        logger.error("Failed to parse box coverage for %s: %s", label_path, exc)
    return ratios

def calculate_aspect_ratio_variance(aspect_ratios: List[float]) -> float:
    """
    Calculate the variance of bounding box aspect ratios.
    Extremely high variance may indicate irregular annotation distributions.
    """
    if len(aspect_ratios) < 2:
        return 0.0
    return float(np.var(aspect_ratios))
