"""
Dataset Augmentation and Balancing Module
==========================================
Modular, reusable functions for augmenting dataset images and labels.
Supports class-balancing by only augmenting minority classes using Albumentations.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

logger = logging.getLogger("augment_dataset")


def build_augmentation_pipeline(cfg: Dict[str, Any]) -> A.Compose:
    """
    Builds the Albumentations Compose pipeline using the specified configuration.
    
    Includes the 7 required transforms:
        HorizontalFlip, VerticalFlip, Rotate, RandomBrightnessContrast,
        GaussianBlur, CLAHE, HueSaturationValue.
    """
    if not HAS_ALBUMENTATIONS:
        raise RuntimeError("albumentations package is required but not installed.")
        
    sp = cfg.get("spatial", {})
    px = cfg.get("pixel", {})
    bl = cfg.get("blur", {})
    
    transforms = [
        A.HorizontalFlip(p=sp.get("horizontal_flip", {}).get("p", 0.5)),
        A.VerticalFlip(p=sp.get("vertical_flip", {}).get("p", 0.5)),
        A.Rotate(
            limit=sp.get("rotate", {}).get("limit", 90),
            border_mode=sp.get("rotate", {}).get("border_mode", 0),
            p=sp.get("rotate", {}).get("p", 0.5)
        ),
        A.RandomBrightnessContrast(
            brightness_limit=px.get("random_brightness_contrast", {}).get("brightness_limit", 0.3),
            contrast_limit=px.get("random_brightness_contrast", {}).get("contrast_limit", 0.3),
            p=px.get("random_brightness_contrast", {}).get("p", 0.5)
        ),
        A.GaussianBlur(
            blur_limit=tuple(bl.get("gaussian_blur", {}).get("blur_limit", [3, 7])),
            p=bl.get("gaussian_blur", {}).get("p", 0.3)
        ),
        A.CLAHE(
            clip_limit=px.get("clahe", {}).get("clip_limit", 4.0),
            tile_grid_size=tuple(px.get("clahe", {}).get("tile_grid_size", [8, 8])),
            p=px.get("clahe", {}).get("p", 0.4)
        ),
        A.HueSaturationValue(
            hue_shift_limit=px.get("hue_saturation_value", {}).get("hue_shift_limit", 15),
            sat_shift_limit=px.get("hue_saturation_value", {}).get("sat_shift_limit", 30),
            val_shift_limit=px.get("hue_saturation_value", {}).get("val_shift_limit", 20),
            p=px.get("hue_saturation_value", {}).get("p", 0.4)
        ),
    ]
    
    min_vis = cfg.get("min_visibility", 0.3)
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=min_vis,
        )
    )


def load_yolo_labels(label_path: Path) -> List[List[float]]:
    """
    Load YOLO labels from file.
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
                    # basic clamping/validation
                    box = [max(0.0, min(1.0, val)) for val in box]
                    if box[2] > 0 and box[3] > 0:
                        boxes.append([cls_id] + box)
    except Exception as exc:
        logger.error("Error reading labels from %s: %s", label_path, exc)
    return boxes


def save_yolo_labels(label_path: Path, boxes: List[List[float]]) -> None:
    """
    Save YOLO labels to file.
    """
    try:
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                cls_id = int(box[0])
                cx, cy, w, h = box[1:]
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    except Exception as exc:
        logger.error("Error writing labels to %s: %s", label_path, exc)


def augment_single_image(
    src_img_path: Path,
    src_lbl_path: Path,
    dst_img_path: Path,
    dst_lbl_path: Path,
    transform: A.Compose,
    jpeg_quality: int
) -> bool:
    """
    Augment a single image-label pair and save it.
    """
    try:
        image = cv2.imread(str(src_img_path))
        if image is None:
            return False
            
        yolo_boxes = load_yolo_labels(src_lbl_path)
        if not yolo_boxes:
            return False
            
        bboxes = [box[1:] for box in yolo_boxes]
        class_labels = [box[0] for box in yolo_boxes]
        
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_img = augmented["image"]
        aug_bboxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]
        
        if not aug_bboxes:
            return False
            
        # Write augmented image
        cv2.imwrite(str(dst_img_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        
        # Write augmented labels
        assembled = [[lbl] + list(bbox) for lbl, bbox in zip(aug_labels, aug_bboxes)]
        save_yolo_labels(dst_lbl_path, assembled)
        return True
    except Exception as exc:
        logger.debug("Augmentation failed for %s: %s", src_img_path.name, exc)
        return False


def balance_dataset_classes(
    train_img_dir: Path,
    train_lbl_dir: Path,
    target_count: int,
    cfg: Dict[str, Any],
    transform: A.Compose,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Balance dataset classes by only augmenting minority classes.
    Uses a ThreadPoolExecutor for concurrent image read, transform, and write.
    """
    # 1. Count class distribution before
    class_counts: Dict[int, int] = {i: 0 for i in range(len(class_names))}
    image_map: Dict[int, List[Path]] = {i: [] for i in range(len(class_names))}
    
    for img_path in train_img_dir.glob("*.jpg"):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        classes_in_img = set()
        for box in boxes:
            cls_id = int(box[0])
            if 0 <= cls_id < len(class_names):
                class_counts[cls_id] += 1
                classes_in_img.add(cls_id)
        for cid in classes_in_img:
            image_map[cid].append(img_path)
            
    if target_count <= 0:
        target_count = max(class_counts.values()) if class_counts else 0
        
    logger.info("Target instances per class: %d", target_count)
    
    output_cfg = cfg.get("output", {})
    file_prefix = output_cfg.get("file_prefix", "aug")
    jpeg_quality = output_cfg.get("jpeg_quality", 95)
    max_per_src = cfg.get("max_aug_per_source", 8)
    
    generated_counts = {i: 0 for i in range(len(class_names))}
    start_time = time.time()
    
    # We will collect tasks to run in parallel
    tasks_to_run = []
    
    for cls_id, count in class_counts.items():
        if count >= target_count:
            logger.info("Class '%s' is already balanced (%d >= %d).", class_names[cls_id], count, target_count)
            continue
            
        candidates = image_map[cls_id]
        if not candidates:
            logger.warning("No candidate images for minority class '%s'", class_names[cls_id])
            continue
            
        deficit = target_count - count
        logger.info("Augmenting class '%s' (current: %d, target: %d, deficit: %d)", class_names[cls_id], count, target_count, deficit)
        
        aug_idx = 0
        while aug_idx < deficit:
            src_img_path = random.choice(candidates)
            src_lbl_path = train_lbl_dir / f"{src_img_path.stem}.txt"
            
            dst_stem = f"{file_prefix}_{src_img_path.stem}_{aug_idx:05d}"
            dst_img_path = train_img_dir / f"{dst_stem}.jpg"
            dst_lbl_path = train_lbl_dir / f"{dst_stem}.txt"
            
            tasks_to_run.append((src_img_path, src_lbl_path, dst_img_path, dst_lbl_path, transform, jpeg_quality, cls_id))
            aug_idx += 1
            
    # Process tasks concurrently in ThreadPoolExecutor
    logger.info("Submitting %d augmentation jobs to ThreadPoolExecutor...", len(tasks_to_run))
    successful_jobs = 0
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for task in tasks_to_run:
            src_img, src_lbl, dst_img, dst_lbl, trans, q, cid = task
            future = executor.submit(augment_single_image, src_img, src_lbl, dst_img, dst_lbl, trans, q)
            futures.append((future, cid))
            
        for future, cid in futures:
            if future.result():
                generated_counts[cid] += 1
                successful_jobs += 1
                
    elapsed = time.time() - start_time
    logger.info("Augmentation balancing done. Total generated files: %d in %.2f seconds.", successful_jobs, elapsed)
    
    # Count distribution after
    after_counts = {i: 0 for i in range(len(class_names))}
    for img_path in train_img_dir.glob("*.jpg"):
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        for box in boxes:
            cls_id = int(box[0])
            if 0 <= cls_id < len(class_names):
                after_counts[cls_id] += 1
                
    return {
        "before": class_counts,
        "after": after_counts,
        "generated": generated_counts,
        "elapsed_seconds": elapsed,
        "target_count": target_count,
    }


def write_balancing_report(
    report_path: Path,
    stats: Dict[str, Any],
    class_names: List[str]
) -> None:
    """
    Generate dataset_balancing_report.md
    """
    before = stats["before"]
    after = stats["after"]
    generated = stats["generated"]
    
    lines = [
        "# Dataset Balancing & Augmentation Report",
        "",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Generation Time**: {stats['elapsed_seconds']:.2f} seconds",
        f"**Target Instance Count**: {stats['target_count']}",
        "",
        "## 1. Class Distribution Before and After Augmentation",
        "",
        "| Class ID | Class Name | Count Before | Count After | Generated | Change (%) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cid in range(len(class_names)):
        name = class_names[cid]
        b = before[cid]
        a = after[cid]
        g = generated[cid]
        pct = ((a - b) / b * 100) if b > 0 else 0
        lines.append(f"| {cid} | {name} | {b} | {a} | {g} | {pct:+.2f}% |")
        
    lines.append("")
    lines.append("## 2. Augmentation Pipelines & Quality Checks")
    lines.append("- Albumentations was configured with standard spatial and pixel transforms.")
    lines.append("- Output JPEG quality was set to 95.")
    lines.append("- Multi-threaded executor was utilized for disk read/write optimization.")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Saved dataset balancing report at %s", report_path)
