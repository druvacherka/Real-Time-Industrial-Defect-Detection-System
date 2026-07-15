"""
Preprocessing Pipeline Benchmark Script
======================================
Measures the computational throughput of resizing and normalization steps
across different interpolation algorithms and configurations.
"""

import time
import logging
from pathlib import Path
import cv2
import numpy as np

from utils.logger import get_logger
from utils.configurable_preprocess import ConfigurablePreprocessor

logger = get_logger("benchmark_preprocessing")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy image for benchmarking (200x200 steel sheet sample representation)
    dummy_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    iterations = 2000
    
    logger.info("Starting preprocessing benchmark metrics...")
    logger.info("Benchmarking with %d iterations on a 300x300 BGR image...", iterations)
    
    # Benchmark Interpolations
    interpolations = ["nearest", "bilinear", "bicubic", "area"]
    interpolation_times = {}
    
    for interp in interpolations:
        interp_enum = ConfigurablePreprocessor.INTERPOLATION_MAP[interp]
        t0 = time.time()
        for _ in range(iterations):
            _ = cv2.resize(dummy_img, (200, 200), interpolation=interp_enum)
        elapsed = time.time() - t0
        interpolation_times[interp] = elapsed
        logger.info("Interpolation: %s - Took %.4f seconds (%.2f FPS)", interp, elapsed, iterations / elapsed)
        
    # Benchmark Normalization Options
    # 1. Min-max scale
    t0 = time.time()
    for _ in range(iterations):
        resized = cv2.resize(dummy_img, (200, 200), interpolation=cv2.INTER_LINEAR)
        _ = resized.astype(np.float32) / 255.0
    min_max_time = time.time() - t0
    
    # 2. Standard score
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t0 = time.time()
    for _ in range(iterations):
        resized = cv2.resize(dummy_img, (200, 200), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        _ = (normalized - mean) / std
    standard_time = time.time() - t0
    
    # Save Report
    report_path = reports_dir / "preprocessing_benchmark.md"
    report_content = f"""# Preprocessing Pipeline Benchmark Report

Generated dynamically by `benchmark_preprocessing.py`.

## ⚙️ Test Specifications
- **Runs per Configuration**: {iterations} iterations
- **Input Dimension**: 300x300x3 BGR array
- **Target Dimension**: 200x200x3

## 📊 Resize Interpolation Speeds
| Interpolation Mode | Total Execution Time (s) | Throughput (FPS) |
| --- | --- | --- |
| Nearest | {interpolation_times['nearest']:.4f} | {iterations / interpolation_times['nearest']:.2f} |
| Bilinear | {interpolation_times['bilinear']:.4f} | {iterations / interpolation_times['bilinear']:.2f} |
| Bicubic | {interpolation_times['bicubic']:.4f} | {iterations / interpolation_times['bicubic']:.2f} |
| Area | {interpolation_times['area']:.4f} | {iterations / interpolation_times['area']:.2f} |

## 🧪 Normalization Modes Comparison
| Normalization Method | Total Execution Time (s) | Throughput (FPS) |
| --- | --- | --- |
| Min-Max Scaling [0, 1] | {min_max_time:.4f} | {iterations / min_max_time:.2f} |
| Standard Z-Score (ImageNet) | {standard_time:.4f} | {iterations / standard_time:.2f} |
"""
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved markdown benchmark report to: %s", report_path)

if __name__ == "__main__":
    main()
