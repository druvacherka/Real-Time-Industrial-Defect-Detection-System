#!/usr/bin/env python3
"""
Dataset Analytics and Visualization Dashboard Generator
======================================================
Real-Time Industrial Defect Detection System

Computes dataset statistics and creates premium visual analytics including class distributions,
split ratios, bounding box densities, average dimensions, aspect ratios, and saves annotated
random image samples with defect overlays inside reports/analytics/.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    IMAGES_DIR,
    LABELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    SPLITS,
    IMAGE_EXTENSIONS,
    DEFECT_CLASSES,
    CLASS_COLORS_BGR,
    ensure_dirs,
)
from utils.dataset_statistics import calculate_stats, load_yolo_labels
from utils.visualization import draw_yolo_annotations

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Setup analytics reporting directory
ANALYTICS_REPORTS_DIR = REPORTS_DIR / "analytics"
ensure_dirs(LOGS_DIR, ANALYTICS_REPORTS_DIR)

# Initialize logger
logger = logging.getLogger("dataset_analytics_dashboard")
logger.setLevel(logging.INFO)

if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "dataset_analytics_dashboard.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def calculate_bbox_dimensions(labels_dir: Path, splits: List[str]) -> Dict[str, Any]:
    """
    Parse YOLO label files to compute box size statistics (width, height, aspect ratio).
    """
    widths = []
    heights = []
    aspect_ratios = []
    
    for split in splits:
        lbl_split_dir = labels_dir / split
        if not lbl_split_dir.exists():
            continue
            
        for lbl_path in lbl_split_dir.glob("*.txt"):
            boxes = load_yolo_labels(lbl_path)
            for box in boxes:
                w, h = box[3], box[4]
                if w > 0 and h > 0:
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
                    
    if not widths:
        return {"mean_width": 0.0, "mean_height": 0.0, "mean_aspect_ratio": 0.0}
        
    return {
        "mean_width": float(np.mean(widths)),
        "mean_height": float(np.mean(heights)),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)),
        "std_width": float(np.std(widths)),
        "std_height": float(np.std(heights)),
    }


def calculate_image_resolutions(images_dir: Path, splits: List[str]) -> Dict[str, Any]:
    """
    Scan image files to compute image resolution statistics.
    """
    resolutions: Dict[str, int] = {}
    total_scanned = 0
    unique_resolutions = set()
    
    for split in splits:
        img_split_dir = images_dir / split
        if not img_split_dir.exists():
            continue
            
        for img_path in img_split_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        res_str = f"{w}x{h}"
                        resolutions[res_str] = resolutions.get(res_str, 0) + 1
                        unique_resolutions.add((w, h))
                        total_scanned += 1
                except Exception as exc:
                    logger.warning("Could not read image %s for resolution check: %s", img_path.name, exc)
                    
    return {
        "total_scanned": total_scanned,
        "resolution_counts": resolutions,
        "unique_resolutions_count": len(unique_resolutions)
    }


def generate_charts(stats: Dict[str, Any], class_names: List[str], output_dir: Path) -> Tuple[Path, Path]:
    """
    Generate class distribution and split pie charts.
    """
    if not HAS_MPL:
        raise RuntimeError("Matplotlib is required for generating charts.")
        
    # 1. Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(class_names))
    overall_counts = [stats["overall"]["class_distribution"][i] for i in range(len(class_names))]
    
    # Custom aesthetic color palette
    bar_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    
    ax.bar(x, overall_counts, color=bar_colors[:len(class_names)], edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, fontsize=10)
    ax.set_ylabel("Bounding Box Counts", fontsize=12)
    ax.set_title("Defect Class Instance Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    dist_chart_path = output_dir / "class_distribution_dashboard.png"
    fig.savefig(dist_chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # 2. Dataset Split Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    split_names = list(stats["splits"].keys())
    split_sizes = [stats["splits"][s]["total_images"] for s in split_names]
    
    pie_colors = ["#2ecc71", "#3498db", "#e74c3c"]
    
    ax.pie(
        split_sizes,
        labels=[s.capitalize() for s in split_names],
        autopct="%1.1f%%",
        colors=pie_colors[:len(split_names)],
        startangle=90,
        wedgeprops={"edgecolor": "black", 'linewidth': 1, 'antialiased': True}
    )
    ax.set_title("Dataset Split Ratio (Images)", fontsize=14, fontweight="bold", pad=15)
    
    split_chart_path = output_dir / "dataset_split_dashboard.png"
    fig.savefig(split_chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return dist_chart_path, split_chart_path


def draw_annotated_samples(
    images_dir: Path,
    labels_dir: Path,
    splits: List[str],
    class_names: List[str],
    colors: List[Tuple[int, int, int]],
    output_dir: Path,
    num_samples_per_split: int = 2
) -> List[str]:
    """
    Select random samples and draw bounding boxes. Return list of paths to annotated images.
    """
    saved_samples = []
    
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        if not img_split_dir.exists():
            continue
            
        img_files = list(img_split_dir.glob("*.jpg"))
        if not img_files:
            continue
            
        samples = random.sample(img_files, min(num_samples_per_split, len(img_files)))
        for img_path in samples:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            lbl_path = lbl_split_dir / f"{img_path.stem}.txt"
            boxes = load_yolo_labels(lbl_path)
            
            # Use premium alpha-blended drawing
            annotated_img = draw_yolo_annotations(
                image=img,
                boxes=boxes,
                class_names=class_names,
                colors=colors,
                line_thickness=2,
                font_scale=0.45,
                fill_alpha=0.25
            )
                
            out_filename = f"sample_{split}_{img_path.stem}.jpg"
            out_path = output_dir / out_filename
            cv2.imwrite(str(out_path), annotated_img)
            saved_samples.append(f"reports/analytics/{out_filename}")
            
    return saved_samples


def write_dashboard_report(
    report_path: Path,
    stats: Dict[str, Any],
    bbox_stats: Dict[str, Any],
    image_res_stats: Dict[str, Any],
    class_names: List[str],
    sample_images: List[str]
) -> None:
    """
    Generate dataset_analytics_report.md
    """
    overall = stats["overall"]
    
    lines = [
        "# Dataset Analytics & Visualisation Dashboard",
        "",
        "This dashboard displays comprehensive visual and numerical analytics computed on the dataset splits.",
        "",
        "## 1. Overall Dataset Metrics Summary",
        "",
        f"- **Total Image Count**: {overall['total_images']}",
        f"- **Total Label Files**: {overall['total_labels']}",
        f"- **Total Bounding Box Instances**: {overall['total_annotations']}",
        f"- **Mean Defect Instances per Image**: {overall['total_annotations'] / max(1, overall['total_images']):.2f}",
        "",
        "## 2. Bounding Box Geometric Analysis",
        "",
        f"- **Average Bounding Box Width**: {bbox_stats['mean_width']:.4f} (relative to image width)",
        f"- **Average Bounding Box Height**: {bbox_stats['mean_height']:.4f} (relative to image height)",
        f"- **Average Aspect Ratio (W/H)**: {bbox_stats['mean_aspect_ratio']:.2f}",
        f"- **Standard Deviation (Width/Height)**: {bbox_stats['std_width']:.4f} / {bbox_stats['std_height']:.4f}",
        "",
        "## 2.5. Image Resolution Analysis",
        "",
        f"- **Total Scanned Images**: {image_res_stats.get('total_scanned', 0)}",
        f"- **Unique Image Resolutions Found**: {image_res_stats.get('unique_resolutions_count', 0)}",
        "**Resolution Breakdown**:",
    ]
    for res_str, count in image_res_stats.get('resolution_counts', {}).items():
        lines.append(f"  - `{res_str}`: {count} images")
    lines.extend([
        "",
        "## 3. Dataset Splits Distribution",
        "",
        "| Split | Image Count | Label Count | Total Annotations | Density (Bboxes/Img) |",
        "| --- | --- | --- | --- | --- |",
    ])
    
    for split, info in stats["splits"].items():
        density = info['total_annotations'] / max(1, info['total_images'])
        lines.append(f"| {split.capitalize()} | {info['total_images']} | {info['total_labels']} | {info['total_annotations']} | {density:.2f} |")
        
    lines.append("")
    lines.append("## 4. Class Distribution & Representation Analysis")
    lines.append("")
    lines.append("| Class ID | Class Name | Total Instances | Percentage (%) | Images Containing Class | Avg Instances/Image | Representation Type |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    
    total_ann = max(1, overall["total_annotations"])
    for cid in range(len(class_names)):
        name = class_names[cid]
        count = overall["class_distribution"][cid]
        pct = (count / total_ann) * 100
        img_count = overall["class_images"][cid]
        avg_density = count / max(1, img_count)
        
        # Simple thresholding logic to print status
        rep_type = "Robust" if pct >= 15 else ("Moderate" if pct >= 10 else "Minority")
        lines.append(f"| {cid} | {name} | {count} | {pct:.2f}% | {img_count} | {avg_density:.2f} | {rep_type} |")
        
    lines.extend([
        "",
        "## 5. Visualizations & Distributions",
        "",
        "### Defect Class Instance Distribution",
        "![Defect Class Distribution](class_distribution_dashboard.png)",
        "",
        "### Dataset Split Ratio",
        "![Dataset Split Ratio](dataset_split_dashboard.png)",
        "",
        "## 6. Sample Ground-Truth Bounding Box Overlays",
        ""
    ])
    
    for sample in sample_images:
        filename = Path(sample).name
        split_name = filename.split("_")[1]
        lines.append(f"### Split: {split_name.upper()} ({filename})")
        lines.append(f"![{filename}]({filename})")
        lines.append("")
        
    lines.append("---")
    lines.append("Dashboard report created automatically by `generate_dataset_dashboard.py`.")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved dataset dashboard report at %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Dataset Analytics and Visualization Dashboard.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ANALYTICS_REPORTS_DIR),
        help="Directory to save the dashboard reports and charts."
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)

    logger.info("Initializing dataset analytics calculation...")
    class_names = [DEFECT_CLASSES[i] for i in sorted(DEFECT_CLASSES.keys())]

    # Step 1: Calculate counts
    stats = calculate_stats(IMAGES_DIR, LABELS_DIR, SPLITS, class_names)

    # Step 2: Bounding box size analytics
    logger.info("Calculating bounding box dimension stats...")
    bbox_stats = calculate_bbox_dimensions(LABELS_DIR, SPLITS)

    # Step 2.5: Image resolution analytics
    logger.info("Scanning image files for resolution reports...")
    image_res_stats = calculate_image_resolutions(IMAGES_DIR, SPLITS)

    # Step 3: Generate visual charts
    logger.info("Generating distribution and split charts...")
    generate_charts(stats, class_names, output_dir)

    # Step 4: Generate sample overlays
    logger.info("Creating ground-truth bounding box overlays on random images...")
    sample_images = draw_annotated_samples(IMAGES_DIR, LABELS_DIR, SPLITS, class_names, CLASS_COLORS_BGR, output_dir)

    # Step 5: Save markdown dashboard report
    report_path = output_dir / "dataset_analytics_report.md"
    write_dashboard_report(report_path, stats, bbox_stats, image_res_stats, class_names, sample_images)

    # Save stats to JSON as well
    stats_json_path = output_dir / "dataset_analytics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "bbox_stats": bbox_stats, "image_res_stats": image_res_stats}, f, indent=2)

    logger.info("Dataset analytics dashboard completed successfully.")


if __name__ == "__main__":
    main()
