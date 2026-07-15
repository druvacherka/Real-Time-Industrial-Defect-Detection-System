"""
Dataset Anomaly Detection Utilities
==================================
Provides functions to detect abnormal image quality, dimension mismatches,
blurriness, and annotation invalidity.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("anomaly_detector")

def detect_image_anomalies(
    image_path: Path,
    target_shape: Tuple[int, int] = (200, 200),
    blur_threshold: float = 100.0,
    brightness_range: Tuple[float, float] = (30.0, 220.0),
    contrast_threshold: float = 15.0
) -> Dict[str, Any]:
    """
    Detect quality and dimension anomalies in a single image.
    
    Args:
        image_path: Path to the image file.
        target_shape: Expected (height, width) of the image.
        blur_threshold: Laplacian variance threshold; lower is blurrier.
        brightness_range: Acceptable range (min, max) for average pixel intensity.
        contrast_threshold: Acceptable minimum standard deviation of pixel intensities.
        
    Returns:
        Dict with anomaly check results.
    """
    anomalies = []
    issues = {}
    
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"corrupted": True, "anomalies": ["Failed to load image via OpenCV"]}
    except Exception as exc:
        return {"corrupted": True, "anomalies": [f"Image reading error: {str(exc)}"]}
        
    h, w = img.shape[:2]
    
    # Dimension check
    if (h, w) != target_shape:
        issues["dimension_mismatch"] = f"Expected {target_shape}, got ({h}, {w})"
        anomalies.append(f"Dimension mismatch: expected {target_shape}, got ({h}, {w})")
        
    # Blurriness check (Laplacian variance)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        issues["blurry"] = f"Laplacian variance {laplacian_var:.2f} < threshold {blur_threshold}"
        anomalies.append(f"Blurry image (variance: {laplacian_var:.2f})")
        
    # Brightness check
    mean_val = np.mean(img)
    if mean_val < brightness_range[0]:
        issues["too_dark"] = f"Mean brightness {mean_val:.2f} < threshold {brightness_range[0]}"
        anomalies.append(f"Image too dark (average: {mean_val:.2f})")
    elif mean_val > brightness_range[1]:
        issues["too_bright"] = f"Mean brightness {mean_val:.2f} > threshold {brightness_range[1]}"
        anomalies.append(f"Image too bright (average: {mean_val:.2f})")
        
    # Contrast check
    std_val = np.std(img)
    if std_val < contrast_threshold:
        issues["low_contrast"] = f"Contrast std {std_val:.2f} < threshold {contrast_threshold}"
        anomalies.append(f"Low contrast (std: {std_val:.2f})")
        
    # Entropy check (information content)
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    if entropy < 3.0:
        issues["low_entropy"] = f"Entropy {entropy:.2f} < 3.0"
        anomalies.append(f"Low information entropy (entropy: {entropy:.2f})")

    return {
        "corrupted": False,
        "width": w,
        "height": h,
        "blur_score": float(laplacian_var),
        "brightness": float(mean_val),
        "contrast": float(std_val),
        "entropy": float(entropy),
        "issues": issues,
        "anomalies": anomalies
    }

def detect_annotation_anomalies(
    label_path: Path,
    valid_class_ids: Set[int]
) -> Dict[str, Any]:
    """
    Scan a YOLO annotation file for parsing errors, coordinates out of bounds,
    and invalid class IDs.
    
    Args:
        label_path: Path to the .txt annotation file.
        valid_class_ids: Set of acceptable class integers.
        
    Returns:
        Dict with status and lists of issues.
    """
    anomalies = []
    issues = {
        "malformed_lines": [],
        "invalid_classes": [],
        "out_of_bounds": []
    }
    
    if not label_path.exists():
        return {"exists": False, "anomalies": ["Label file does not exist"]}
        
    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        return {"corrupted": True, "anomalies": [f"Could not read label: {str(exc)}"]}
        
    if not content:
        return {"exists": True, "empty": True, "anomalies": []}
        
    lines = content.splitlines()
    for idx, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
            
        if len(parts) != 5:
            msg = f"Line {idx}: Expected 5 values, got {len(parts)} - '{line}'"
            issues["malformed_lines"].append(msg)
            anomalies.append(msg)
            continue
            
        try:
            cls_str, cx_str, cy_str, w_str, h_str = parts
            cls_id = int(float(cls_str))
            cx = float(cx_str)
            cy = float(cy_str)
            bw = float(w_str)
            bh = float(h_str)
        except ValueError as exc:
            msg = f"Line {idx}: Parsing failure: {str(exc)} - '{line}'"
            issues["malformed_lines"].append(msg)
            anomalies.append(msg)
            continue
            
        # Class check
        if cls_id not in valid_class_ids:
            msg = f"Line {idx}: Invalid class ID {cls_id} (Expected in {valid_class_ids})"
            issues["invalid_classes"].append(msg)
            anomalies.append(msg)
            
        # Bounding box sanity checks
        box_issues = []
        if any(math.isnan(v) or math.isinf(v) for v in (cx, cy, bw, bh)):
            box_issues.append("NaN/Inf coordinates")
        else:
            if not (0.0 <= cx <= 1.0):
                box_issues.append(f"cx={cx} out of [0, 1]")
            if not (0.0 <= cy <= 1.0):
                box_issues.append(f"cy={cy} out of [0, 1]")
            if not (0.0 < bw <= 1.0):
                box_issues.append(f"w={bw} out of (0, 1]")
            if not (0.0 < bh <= 1.0):
                box_issues.append(f"h={bh} out of (0, 1]")
                
        if box_issues:
            msg = f"Line {idx}: Sanity issues: {', '.join(box_issues)}"
            issues["out_of_bounds"].append(msg)
            anomalies.append(msg)
            
    return {
        "exists": True,
        "empty": False,
        "issues": issues,
        "anomalies": anomalies
    }
