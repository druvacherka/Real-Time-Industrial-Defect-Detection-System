"""
Dataset Anomaly Detection Script
===============================
Runs automated checks over image and label files across all splits.
Generates anomaly detection reports inside reports/anomalies/.
"""

import json
import logging
from pathlib import Path
from utils.logger import get_logger
from utils.anomaly_detector import detect_image_anomalies, detect_annotation_anomalies

logger = get_logger("detect_anomalies_script")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "dataset"
    yolo_dir = dataset_dir / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    
    reports_dir = root_dir / "reports" / "anomalies"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    valid_classes = {0, 1, 2, 3, 4, 5}
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    logger.info("Starting dataset anomaly detection scan...")
    logger.info("Dataset root: %s", dataset_dir)
    
    report_data = {
        "summary": {
            "total_images_scanned": 0,
            "total_labels_scanned": 0,
            "total_image_anomalies": 0,
            "total_label_anomalies": 0,
            "corrupted_images": 0
        },
        "anomalous_images": [],
        "anomalous_labels": []
    }
    
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        if not img_split_dir.exists():
            logger.warning("Images directory missing for split: %s", split)
            continue
            
        logger.info("Scanning split: %s", split)
        
        # Scan images
        image_files = [f for f in img_split_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        for img_path in image_files:
            report_data["summary"]["total_images_scanned"] += 1
            res = detect_image_anomalies(img_path)
            if res.get("corrupted") or res.get("anomalies"):
                report_data["summary"]["total_image_anomalies"] += len(res.get("anomalies", []))
                if res.get("corrupted"):
                    report_data["summary"]["corrupted_images"] += 1
                report_data["anomalous_images"].append({
                    "split": split,
                    "filename": img_path.name,
                    "anomalies": res.get("anomalies"),
                    "metrics": {
                        "width": res.get("width"),
                        "height": res.get("height"),
                        "blur_score": res.get("blur_score"),
                        "brightness": res.get("brightness"),
                        "contrast": res.get("contrast"),
                        "entropy": res.get("entropy")
                    }
                })
                
        # Scan labels
        if lbl_split_dir.exists():
            label_files = [f for f in lbl_split_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
            for lbl_path in label_files:
                report_data["summary"]["total_labels_scanned"] += 1
                res = detect_annotation_anomalies(lbl_path, valid_classes)
                if res.get("anomalies"):
                    report_data["summary"]["total_label_anomalies"] += len(res.get("anomalies"))
                    report_data["anomalous_labels"].append({
                        "split": split,
                        "filename": lbl_path.name,
                        "anomalies": res.get("anomalies")
                    })
                    
    # Save JSON Report
    json_path = reports_dir / "anomaly_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    logger.info("JSON report saved to %s", json_path)
    
    # Save Markdown Report
    md_path = reports_dir / "anomaly_report.md"
    md_content = f"""# Dataset Anomaly Detection Report

Generated dynamically by `detect_anomalies.py`.

## 📊 Summary Statistics
- **Total Images Scanned**: {report_data["summary"]["total_images_scanned"]}
- **Total Labels Scanned**: {report_data["summary"]["total_labels_scanned"]}
- **Corrupted Images**: {report_data["summary"]["corrupted_images"]}
- **Image Anomalies Detected**: {report_data["summary"]["total_image_anomalies"]}
- **Label/Annotation Anomalies Detected**: {report_data["summary"]["total_label_anomalies"]}

## 🖼️ Anomalous Images Details (Top 20)
| Split | Filename | Anomalies | Stats (Blur / Brightness / Contrast / Entropy) |
| --- | --- | --- | --- |
"""
    for item in report_data["anomalous_images"][:20]:
        stats = item["metrics"]
        details = f"Blur: {stats['blur_score']:.1f}, Bright: {stats['brightness']:.1f}, Contrast: {stats['contrast']:.1f}, Entropy: {stats['entropy']:.2f}"
        anomalies_str = ", ".join(item["anomalies"])
        md_content += f"| {item['split']} | {item['filename']} | {anomalies_str} | {details} |\n"
        
    md_content += f"""
## 🏷️ Anomalous Labels Details (Top 20)
| Split | Filename | Anomalies |
| --- | --- | --- |
"""
    for item in report_data["anomalous_labels"][:20]:
        anomalies_str = "; ".join(item["anomalies"])
        md_content += f"| {item['split']} | {item['filename']} | {anomalies_str} |\n"
        
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Markdown report saved to %s", md_path)

if __name__ == "__main__":
    main()
