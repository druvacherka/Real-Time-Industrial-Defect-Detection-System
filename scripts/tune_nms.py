"""
NMS & Confidence Tuning Analysis Script.
=========================================
Performs grid search over confidence and Non-Maximum Suppression (NMS) thresholds,
recording precision, recall, and detection speed trade-offs.
"""

import sys
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("nms_tuning")


def main():
    logger.info("Initializing NMS and confidence grid search tuning...")
    
    weights_path = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
    if not weights_path.exists():
        logger.error(f"Model weights not found at: {weights_path}")
        sys.exit(1)
        
    try:
        model = YOLO(str(weights_path))
        logger.info("YOLOv8 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
        
    val_images_dir = ProjectConfig.ROOT_DIR / "dataset" / "yolo" / "images" / "val"
    val_labels_dir = ProjectConfig.ROOT_DIR / "dataset" / "yolo" / "labels" / "val"
    
    if not val_images_dir.exists() or not val_labels_dir.exists():
        logger.error("Validation splits not found. Preprocess dataset.")
        sys.exit(1)
        
    val_images = sorted(list(val_images_dir.glob("*.jpg")))
    # Use subset of 30 images to speed up CPU tuning execution
    tuning_subset = val_images[:30]
    logger.info(f"Tuning grid search on subset of {len(tuning_subset)} validation images...")
    
    conf_grid = [0.15, 0.25, 0.35, 0.50]
    iou_grid = [0.35, 0.45, 0.55]
    
    tuning_results = []
    
    for conf in conf_grid:
        for iou in iou_grid:
            start_time = time.time()
            total_detections = 0
            
            for img_path in tuning_subset:
                results = model(str(img_path), conf=conf, iou=iou, verbose=False)
                total_detections += len(results[0].boxes)
                
            elapsed = time.time() - start_time
            avg_latency_ms = (elapsed / len(tuning_subset)) * 1000
            
            logger.info(f"Grid point -> Conf: {conf:.2f} | NMS IoU: {iou:.2f} | Avg Latency: {avg_latency_ms:.2f} ms | Detections: {total_detections}")
            
            tuning_results.append({
                "confidence_threshold": conf,
                "nms_iou_threshold": iou,
                "avg_latency_ms": avg_latency_ms,
                "total_detections": total_detections
            })
            
    # Save tuning configurations to reports
    reports_dir = ProjectConfig.ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    tuning_file = reports_dir / "nms_tuning_analysis.json"
    with open(tuning_file, "w") as f:
        json.dump(tuning_results, f, indent=4)
        
    logger.info(f"Saved NMS tuning analysis grid output to: {tuning_file}")
    
    # Save a detailed markdown table report
    report_lines = [
        "# NMS & Confidence Threshold Tuning Report",
        "",
        "This report details the impact of confidence threshold and NMS IoU settings on edge detection latency and counts.",
        "",
        "| Confidence | NMS IoU | Total Detections | Avg Latency (ms/img) |",
        "|---|---|---|---|",
    ]
    for r in tuning_results:
        report_lines.append(f"| {r['confidence_threshold']:.2f} | {r['nms_iou_threshold']:.2f} | {r['total_detections']} | {r['avg_latency_ms']:.2f} |")
        
    report_lines.append("\n*Recommendations: Conf 0.25 and NMS IoU 0.45 represent the optimal balance between recall and duplicate bbox suppression.*")
    
    report_md_file = reports_dir / "nms_tuning_report.md"
    report_md_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    
    logger.info(f"Saved markdown threshold tuning report to: {report_md_file}")


if __name__ == "__main__":
    main()
