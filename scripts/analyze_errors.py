"""
Error analysis script to parse validation labels, match with predictions,
log False Positives (FP) & False Negatives (FN), and generate a confusion matrix.
==================================================================================
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("error_analysis")

DEFECT_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches"
}


def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def yolo_to_xyxy(yolo_box, img_w, img_h):
    """Converts normalized YOLO format [x_center, y_center, w, h] to pixel [x1, y1, x2, y2]."""
    xc, yc, w, h = yolo_box
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def main():
    logger.info("Starting model error analysis and confusion matrix generation...")
    
    weights_path = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
    if not weights_path.exists():
        logger.error(f"Trained model weights not found at: {weights_path}")
        sys.exit(1)
        
    try:
        model = YOLO(str(weights_path))
        logger.info("Custom model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
        
    val_images_dir = ProjectConfig.ROOT_DIR / "dataset" / "yolo" / "images" / "val"
    val_labels_dir = ProjectConfig.ROOT_DIR / "dataset" / "yolo" / "labels" / "val"
    
    if not val_images_dir.exists() or not val_labels_dir.exists():
        logger.error("Validation dataset directories not found. Verify preprocessed dataset splits.")
        sys.exit(1)
        
    label_files = list(val_labels_dir.glob("*.txt"))
    logger.info(f"Found {len(label_files)} validation labels to analyze.")
    
    # 6 classes + 1 background class (index 6 for background FP/FN)
    num_classes = len(DEFECT_CLASSES)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
    
    fp_details = []
    fn_details = []
    
    # Iterate through each validation image and label file
    for lbl_file in label_files:
        img_name = lbl_file.stem
        # Check image file formats
        img_file = None
        for ext in [".jpg", ".png", ".jpeg"]:
            temp = val_images_dir / f"{img_name}{ext}"
            if temp.exists():
                img_file = temp
                break
                
        if not img_file:
            logger.warning(f"Could not find matching image for label: {lbl_file.name}")
            continue
            
        # Get image dimensions (NEU-DET dataset images are typically 200x200)
        with Image.open(img_file) as img:
            img_w, img_h = img.size
            
        # 1. Parse Ground Truth Boxes
        gt_boxes = []
        if lbl_file.stat().st_size > 0:
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        yolo_box = [float(x) for x in parts[1:]]
                        pixel_box = yolo_to_xyxy(yolo_box, img_w, img_h)
                        gt_boxes.append({"class_id": cls_id, "box": pixel_box, "matched": False})
                        
        # 2. Get Model Predictions
        try:
            preds = model(str(img_file), verbose=False)[0]
            pred_boxes = []
            for box in preds.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [float(x) for x in box.xyxy[0].tolist()]
                pred_boxes.append({"class_id": cls_id, "conf": conf, "box": xyxy, "matched": False})
        except Exception as pred_err:
            logger.error(f"Error predicting image {img_file.name}: {pred_err}")
            continue
            
        # 3. Match Predictions to Ground Truths (descending confidence order)
        pred_boxes.sort(key=lambda x: x["conf"], reverse=True)
        
        for p in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for idx, gt in enumerate(gt_boxes):
                if gt["matched"]:
                    continue
                iou = calculate_iou(p["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                    
            # Match threshold of 0.45 IoU
            if best_gt_idx != -1 and best_iou >= 0.45:
                gt_boxes[best_gt_idx]["matched"] = True
                p["matched"] = True
                
                gt_cls = gt_boxes[best_gt_idx]["class_id"]
                pred_cls = p["class_id"]
                
                # Confusion matrix update
                conf_matrix[gt_cls, pred_cls] += 1
                
                if gt_cls != pred_cls:
                    # Misclassification logs (counts as FP for pred_cls and FN for gt_cls)
                    fp_details.append({
                        "image": img_file.name,
                        "ground_truth": DEFECT_CLASSES[gt_cls],
                        "predicted": DEFECT_CLASSES[pred_cls],
                        "confidence": p["conf"],
                        "box": p["box"],
                        "type": "misclass"
                    })
            else:
                # Unmatched prediction -> False Positive (predicted background as defect)
                conf_matrix[num_classes, p["class_id"]] += 1  # background row, predicted col
                fp_details.append({
                    "image": img_file.name,
                    "ground_truth": "background",
                    "predicted": DEFECT_CLASSES[p["class_id"]],
                    "confidence": p["conf"],
                    "box": p["box"],
                    "type": "background_fp"
                })
                
        # 4. Record Unmatched Ground Truths -> False Negatives (model missed defect)
        for gt in gt_boxes:
            if not gt["matched"]:
                conf_matrix[gt["class_id"], num_classes] += 1  # gt row, background col
                fn_details.append({
                    "image": img_file.name,
                    "ground_truth": DEFECT_CLASSES[gt["class_id"]],
                    "predicted": "background",
                    "box": gt["box"]
                })
                
    # Generate detailed confusion matrix report and FP/FN details logs
    results_dir = ProjectConfig.ROOT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = results_dir / "error_analysis_report.md"
    logger.info(f"Generating error analysis and confusion matrix report: {report_path}")
    
    # Format Confusion Matrix
    matrix_lines = []
    matrix_lines.append("| GT \\ Pred | " + " | ".join([DEFECT_CLASSES[i] for i in range(num_classes)]) + " | background |")
    matrix_lines.append("|---| " + " | ".join(["---" for _ in range(num_classes + 1)]) + " |")
    for i in range(num_classes + 1):
        row_name = DEFECT_CLASSES[i] if i < num_classes else "background"
        row_vals = [str(conf_matrix[i, j]) for j in range(num_classes + 1)]
        matrix_lines.append(f"| **{row_name}** | " + " | ".join(row_vals) + " |")
        
    # Analyze total metrics
    total_fps = len(fp_details)
    total_fns = len(fn_details)
    
    report_content = f"""# Confusion Matrix and Error Analysis Report

This report documents the baseline model's classification errors, detailing False Positives, False Negatives, and Misclassifications on the Validation split.

---

## 📊 Confusion Matrix

The table below displays the prediction-to-ground-truth mapping. Rows represent Ground Truth classes and columns represent Predicted classes.

{"\n".join(matrix_lines)}

*Note: background column represents False Negatives (model missed defects); background row represents False Positives (model detected defects on background).*

---

## 🔍 Error Analysis Summary

* **Total False Positives (FP)**: {total_fps}
* **Total False Negatives (FN)**: {total_fns}

### False Positives / Negatives Count by Class

| Class Name | False Positives (Background FP) | False Negatives (Model Missed) |
|---|---|---|
"""
    
    # Calculate counts per class
    for cid, name in DEFECT_CLASSES.items():
        fp_count = sum(1 for x in fp_details if x["predicted"] == name and x["ground_truth"] == "background")
        fn_count = sum(1 for x in fn_details if x["ground_truth"] == name)
        report_content += f"| {name} | {fp_count} | {fn_count} |\n"
        
    report_content += """
---

## 📋 Detailed Error Log Samples (First 15 Errors)

The table below lists individual False Positive and False Negative defect details for debugging.

| Image File | Ground Truth | Predicted | Confidence | Bounding Box [x1, y1, x2, y2] |
|---|---|---|---|---|
"""
    
    # Log samples
    sample_errors = []
    # Merge FP and FN details for reporting
    for fp in fp_details[:10]:
        sample_errors.append(f"| {fp['image']} | {fp['ground_truth']} | {fp['predicted']} | {fp['confidence']:.4f} | {list(np.round(fp['box'], 1))} |")
    for fn in fn_details[:10]:
        sample_errors.append(f"| {fn['image']} | {fn['ground_truth']} | {fn['predicted']} | N/A | {list(np.round(fn['box'], 1))} |")
        
    report_content += "\n".join(sample_errors[:15]) + "\n"
    
    report_path.write_text(report_content, encoding="utf-8")
    
    # Also log to a standard text file results/error_analysis.log
    log_path = results_dir / "error_analysis.log"
    log_lines = []
    for fp in fp_details:
        log_lines.append(f"FP | {fp['image']} | GT: {fp['ground_truth']} -> PRED: {fp['predicted']} (Conf: {fp['confidence']:.4f}) | Box: {fp['box']}")
    for fn in fn_details:
        log_lines.append(f"FN | {fn['image']} | GT: {fn['ground_truth']} -> PRED: background | Box: {fn['box']}")
        
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    
    logger.info(f"Confusion matrix report generated successfully. Logged {total_fps} FPs and {total_fns} FNs.")


if __name__ == "__main__":
    main()
