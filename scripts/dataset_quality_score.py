"""
Dataset Quality Scoring Script
==============================
Analyzes dataset resolution, blurriness, annotation coverage, and class balance.
Generates charts and saves a markdown quality report.
"""

import logging
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.logger import get_logger
from utils.quality_scoring import calculate_shannon_entropy, compute_quality_score, parse_box_coverage

logger = get_logger("dataset_quality_score")

def main():
    root_dir = Path(__file__).resolve().parent.parent
    dataset_dir = root_dir / "dataset"
    yolo_dir = dataset_dir / "yolo"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = reports_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    logger.info("Initializing dataset quality scoring analysis...")
    
    class_counts = [0] * 6
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
    
    box_areas: List[float] = []
    resolutions = []
    blur_scores = []
    total_error_annotations = 0
    total_annotations = 0
    
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        if not img_split_dir.exists():
            continue
            
        logger.info("Analyzing split: %s", split)
        
        image_files = [f for f in img_split_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
        for img_path in image_files:
            # Resolution & blur
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    h, w = img.shape[:2]
                    resolutions.append((w, h))
                    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
                    blur_scores.append(lap_var)
            except Exception as exc:
                logger.warning("Could not read image %s: %s", img_path.name, exc)
                
            # Labels
            lbl_path = lbl_split_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                areas = parse_box_coverage(lbl_path)
                box_areas.extend(areas)
                
                try:
                    content = lbl_path.read_text(encoding="utf-8").strip()
                    if content:
                        for line in content.splitlines():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                total_annotations += 1
                                try:
                                    cls_id = int(float(parts[0]))
                                    if 0 <= cls_id < 6:
                                        class_counts[cls_id] += 1
                                    else:
                                        total_error_annotations += 1
                                except ValueError:
                                    total_error_annotations += 1
                            else:
                                total_error_annotations += 1
                except Exception:
                    pass
                    
    # Computations
    total_imgs = len(resolutions)
    avg_w = np.mean([r[0] for r in resolutions]) if resolutions else 0
    avg_h = np.mean([r[1] for r in resolutions]) if resolutions else 0
    
    # Class Balance Analysis (Shannon Entropy / Uniform Entropy for 6 classes = log2(6) = 2.585)
    entropy = calculate_shannon_entropy(class_counts)
    max_entropy = np.log2(6)
    balance_score = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Blur score normalize (average Laplacian variance in [0, 1] scaling with 500 max)
    avg_blur = np.mean(blur_scores) if blur_scores else 0.0
    normalized_blur = min(1.0, avg_blur / 500.0)
    
    # Resolution score (expected 200x200)
    resolution_score = 1.0 if abs(avg_w - 200.0) < 5 and abs(avg_h - 200.0) < 5 else 0.8
    
    error_rate = total_error_annotations / total_annotations if total_annotations > 0 else 0.0
    
    composite_score = compute_quality_score(
        resolution_score=resolution_score,
        blur_score=normalized_blur,
        balance_score=balance_score,
        annotation_error_rate=error_rate
    )
    
    logger.info("Composite Dataset Quality Score: %.2f / 100.0", composite_score)
    
    # Plot Class Balance Chart
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, class_counts, color=["#3498db", "#9b59b6", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
    plt.title("NEU-DET Class Distribution (Augmented)")
    plt.xlabel("Defect Class")
    plt.ylabel("BBox Annotation Count")
    plt.tight_layout()
    chart_path = graphs_dir / "class_balance.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info("Saved class balance graph to %s", chart_path)
    
    # Write Markdown Quality Report
    report_path = reports_dir / "dataset_quality_report.md"
    report_content = f"""# Dataset Quality Scoring and Benchmarking Report

Generated dynamically by `dataset_quality_score.py`.

## 🏆 Overall Dataset Quality Score
**Composite Quality Index**: `{composite_score:.2f} / 100.0`

## 📊 Quality Scoring Component Details
| Quality Dimension | Value / Statistic | Calculated Sub-score | Weight in Index |
| --- | --- | --- | --- |
| **Image Resolution Sanity** | {avg_w:.1f}x{avg_h:.1f} mean dimension | {resolution_score * 25.0:.1f} / 25.0 | 25% |
| **Defect Blur Indicator** | {avg_blur:.1f} average Laplacian variance | {normalized_blur * 25.0:.1f} / 25.0 | 25% |
| **Shannon Class Balance** | {entropy:.3f} entropy (Perfect is 2.585) | {balance_score * 30.0:.1f} / 30.0 | 30% |
| **Annotation Integrity** | {error_rate * 100.0:.2f}% error rate | Penalty: -{error_rate * 100.0 * 2.0:.1f} | Penalty (up to -20) |

## 📐 Annotation Coverage Statistics
- **Total Bounding Boxes Checked**: {total_annotations}
- **Average Box Coverage (Box area / Image area)**: {np.mean(box_areas) * 100.0:.2f}%
- **Median Box Coverage**: {np.median(box_areas) * 100.0:.2f}% if box_areas else 0.0

## 📦 Resolution and Volume Distribution
- **Total Images Analyzed**: {total_imgs}
- **Class Counts**:
"""
    for name, cnt in zip(class_names, class_counts):
        report_content += f"  - **{name}**: {cnt} annotations\n"
        
    report_content += f"""
![Class Distribution Graphs](graphs/class_balance.png)
"""
    report_path.write_text(report_content, encoding="utf-8")
    logger.info("Saved quality markdown report to: %s", report_path)

if __name__ == "__main__":
    main()
