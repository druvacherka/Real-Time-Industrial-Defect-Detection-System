"""
Dataset Statistics and Visualizations Module
=============================================
Modular, reusable functions for calculating dataset statistics and generating
visualization graphs (class distribution, split ratios, and annotated sample images).
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logger = logging.getLogger("dataset_statistics")


def load_yolo_labels(label_path: Path) -> List[List[float]]:
    """
    Read a YOLO label file and return parsed values.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(float(parts[0]))
                    box = [float(x) for x in parts[1:]]
                    boxes.append([cls_id] + box)
    except Exception as exc:
        logger.error("Error reading label file %s: %s", label_path, exc)
    return boxes


def calculate_stats(
    images_dir: Path,
    labels_dir: Path,
    splits: List[str],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate statistics including:
      - Total images and labels
      - Instances per class (both overall and per-split)
      - Images per class (both overall and per-split)
      - Class distribution percentages
    """
    stats = {
        "splits": {},
        "overall": {
            "total_images": 0,
            "total_labels": 0,
            "total_annotations": 0,
            "class_distribution": {i: 0 for i in range(len(class_names))},
            "class_images": {i: 0 for i in range(len(class_names))},
        }
    }
    
    overall_images_set: Dict[int, Set[str]] = {i: set() for i in range(len(class_names))}
    
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        img_files = list(img_split_dir.glob("*.jpg"))
        lbl_files = list(lbl_split_dir.glob("*.txt"))
        
        split_counts = {i: 0 for i in range(len(class_names))}
        split_images_set: Dict[int, Set[str]] = {i: set() for i in range(len(class_names))}
        
        total_annotations = 0
        for lbl_path in lbl_files:
            boxes = load_yolo_labels(lbl_path)
            total_annotations += len(boxes)
            seen_classes = set()
            for box in boxes:
                cls_id = int(box[0])
                if 0 <= cls_id < len(class_names):
                    split_counts[cls_id] += 1
                    seen_classes.add(cls_id)
            for cls_id in seen_classes:
                split_images_set[cls_id].add(lbl_path.stem)
                overall_images_set[cls_id].add(f"{split}/{lbl_path.stem}")
                
        stats["splits"][split] = {
            "total_images": len(img_files),
            "total_labels": len(lbl_files),
            "total_annotations": total_annotations,
            "class_distribution": split_counts,
            "class_images": {cid: len(st) for cid, st in split_images_set.items()}
        }
        
        stats["overall"]["total_images"] += len(img_files)
        stats["overall"]["total_labels"] += len(lbl_files)
        stats["overall"]["total_annotations"] += total_annotations
        for i in range(len(class_names)):
            stats["overall"]["class_distribution"][i] += split_counts[i]
            
    for i in range(len(class_names)):
        stats["overall"]["class_images"][i] = len(overall_images_set[i])
        
    return stats


def generate_visualizations(
    stats: Dict[str, Any],
    vis_dir: Path,
    class_names: List[str],
    images_dir: Path,
    labels_dir: Path,
    splits: List[str],
    colors: List[Tuple[int, int, int]]
) -> None:
    """
    Generate all required charts and sample bbox visualizations:
      - Class distribution bar chart
      - Dataset split pie chart
      - Annotated random image samples saved to vis_dir
    """
    if not HAS_MPL:
        logger.warning("Matplotlib not available. Skipping chart generation.")
        return
        
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names))
    overall_counts = [stats["overall"]["class_distribution"][i] for i in range(len(class_names))]
    
    ax.bar(x, overall_counts, color="#3498db", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylabel("Instance Count")
    ax.set_title("Defect Class Distribution (Total Instances)")
    fig.savefig(vis_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Dataset Split Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    split_names = list(stats["splits"].keys())
    split_sizes = [stats["splits"][s]["total_images"] for s in split_names]
    
    ax.pie(split_sizes, labels=[s.capitalize() for s in split_names], autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c", "#f1c40f"], startangle=90)
    ax.set_title("Dataset Split Ratio (Images)")
    fig.savefig(vis_dir / "dataset_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 3. Draw Bounding Boxes on Random Image Samples
    logger.info("Drawing annotated samples...")
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        img_files = list(img_split_dir.glob("*.jpg"))
        if not img_files:
            continue
            
        # Select 2 random samples
        samples = random.sample(img_files, min(2, len(img_files)))
        for idx, img_path in enumerate(samples):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w, _ = img.shape
            lbl_path = lbl_split_dir / f"{img_path.stem}.txt"
            boxes = load_yolo_labels(lbl_path)
            
            for box in boxes:
                cls_id = int(box[0])
                cx, cy, bw, bh = box[1:]
                
                # Convert YOLO to pixel coordinates
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                color = colors[cls_id % len(colors)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                name = class_names[cls_id]
                cv2.putText(img, name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
            out_path = vis_dir / f"sample_{split}_{idx}.jpg"
            cv2.imwrite(str(out_path), img)


def write_statistics_report(
    report_path: Path,
    stats: Dict[str, Any],
    class_names: List[str]
) -> None:
    """
    Generate dataset_statistics_report.md
    """
    overall = stats["overall"]
    
    lines = [
        "# Dataset Statistics Report",
        "",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Overall Statistics Summary",
        "",
        f"- **Total Image Files**: {overall['total_images']}",
        f"- **Total Label Files**: {overall['total_labels']}",
        f"- **Total Bounding Box Instances**: {overall['total_annotations']}",
        "",
        "## 2. Per-Split Breakdown",
        "",
        "| Split | Image Files | Label Files | Annotation Instances |",
        "| --- | --- | --- | --- |",
    ]
    
    for split, info in stats["splits"].items():
        lines.append(f"| {split.capitalize()} | {info['total_images']} | {info['total_labels']} | {info['total_annotations']} |")
        
    lines.append("")
    lines.append("## 3. Instance Distribution per Class")
    lines.append("")
    lines.append("| Class ID | Class Name | Total Instances | Percentage (%) | Images Containing Class |")
    lines.append("| --- | --- | --- | --- | --- |")
    
    total_ann = max(1, overall["total_annotations"])
    for cid in range(len(class_names)):
        name = class_names[cid]
        count = overall["class_distribution"][cid]
        pct = (count / total_ann) * 100
        img_count = overall["class_images"][cid]
        lines.append(f"| {cid} | {name} | {count} | {pct:.2f}% | {img_count} |")
        
    lines.append("")
    lines.append("## 4. Visualizations Index")
    lines.append("- Class Distribution Bar Chart: `reports/visualizations/class_distribution.png`")
    lines.append("- Dataset Split Pie Chart: `reports/visualizations/dataset_split.png`")
    lines.append("- Visualized annotated samples can be found in `reports/visualizations/sample_*.jpg`")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved statistics report at %s", report_path)
