"""
Dataset Validation Module
=========================
Modular, reusable verification functions for the NEU Metal Surface Defects dataset.
Contains folder structure checking, image corruption tests, duplicate checks,
class ID validations, and bounding box checks.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2

logger = logging.getLogger("dataset_verification")

def validate_folder_structure(
    dataset_root: Path,
    splits: List[str]
) -> Dict[str, str]:
    """
    Verify that all required dataset directories exist.
    
    Args:
        dataset_root: Path to the dataset root.
        splits: List of dataset splits (e.g. ['train', 'val', 'test']).
        
    Returns:
        Dict mapping directory relative path to status ("EXISTS" or "MISSING").
    """
    results = {}
    yolo_root = dataset_root / "yolo"
    images_dir = yolo_root / "images"
    labels_dir = yolo_root / "labels"
    
    required_dirs = [dataset_root, yolo_root, images_dir, labels_dir]
    for s in splits:
        required_dirs.append(images_dir / s)
        required_dirs.append(labels_dir / s)
        
    for p in required_dirs:
        try:
            rel = p.relative_to(dataset_root.parent)
        except ValueError:
            rel = p
        
        status = "EXISTS" if p.exists() else "MISSING"
        results[str(rel)] = status
        if status == "MISSING":
            logger.warning("Required directory missing: %s", rel)
        else:
            logger.debug("Required directory exists: %s", rel)
            
    return results


def check_pairs_and_empty_files(
    images_dir: Path,
    labels_dir: Path,
    splits: List[str],
    image_extensions: Set[str]
) -> Dict[str, Any]:
    """
    Check image-label matching and identify empty label files.
    
    Returns:
        Dict detailing missing labels, missing images, and empty label files per split.
    """
    report = {
        "missing_labels": {},
        "missing_images": {},
        "empty_files": {},
    }
    
    for split in splits:
        img_split_dir = images_dir / split
        lbl_split_dir = labels_dir / split
        
        if not img_split_dir.exists() or not lbl_split_dir.exists():
            continue
            
        img_stems = {
            f.stem: f for f in img_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        }
        lbl_stems = {
            f.stem: f for f in lbl_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }
        
        # Missing labels
        no_label = sorted(img_stems.keys() - lbl_stems.keys())
        if no_label:
            report["missing_labels"][split] = no_label
            
        # Missing images
        no_image = sorted(lbl_stems.keys() - img_stems.keys())
        if no_image:
            report["missing_images"][split] = no_image
            
        # Empty annotations
        empty = []
        for stem, path in lbl_stems.items():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if not content:
                    empty.append(path.name)
            except Exception as exc:
                logger.error("Failed to read label file %s: %s", path, exc)
                empty.append(path.name)
                
        if empty:
            report["empty_files"][split] = empty
            
    return report


def check_corrupted_images(
    images_dir: Path,
    splits: List[str],
    image_extensions: Set[str]
) -> Dict[str, List[str]]:
    """
    Scan all image files using OpenCV to identify decoding/corruption errors.
    """
    corrupted = {}
    for split in splits:
        img_split_dir = images_dir / split
        if not img_split_dir.exists():
            continue
            
        split_corrupted = []
        image_files = [
            f for f in img_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("cv2.imread returned None")
                if img.shape[0] == 0 or img.shape[1] == 0:
                    raise ValueError("Zero-dimension image detected")
            except Exception as exc:
                logger.warning("Corrupted image found: %s/%s - %s", split, img_path.name, exc)
                split_corrupted.append(img_path.name)
                
        if split_corrupted:
            corrupted[split] = split_corrupted
            
    return corrupted


def check_duplicate_images(
    images_dir: Path,
    splits: List[str],
    image_extensions: Set[str]
) -> List[Dict[str, Any]]:
    """
    Identify duplicate images based on MD5 checksum.
    """
    hash_map: Dict[str, List[str]] = {}
    
    for split in splits:
        img_split_dir = images_dir / split
        if not img_split_dir.exists():
            continue
            
        for img_path in img_split_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                try:
                    digest = hashlib.md5(img_path.read_bytes()).hexdigest()
                    hash_map.setdefault(digest, []).append(f"{split}/{img_path.name}")
                except Exception as exc:
                    logger.error("Could not compute hash for %s: %s", img_path, exc)
                    
    duplicates = [
        {"md5": md5, "files": sorted(paths)}
        for md5, paths in hash_map.items()
        if len(paths) > 1
    ]
    return duplicates


def validate_annotation_consistency(
    labels_dir: Path,
    splits: List[str],
    valid_class_ids: Set[int]
) -> Dict[str, Any]:
    """
    Check label format, number of elements, numeric type conversion,
    valid class IDs, and bounding-box coordinates constraint in [0, 1].
    """
    report = {
        "invalid_class_ids": {},
        "out_of_range_coords": {},
        "malformed_lines": {},
        "class_distribution": {},
        "total_annotations_checked": 0,
        "total_errors": 0,
    }
    
    class_distribution: Dict[int, int] = {}
    
    for split in splits:
        lbl_split_dir = labels_dir / split
        if not lbl_split_dir.exists():
            continue
            
        invalid_classes = []
        out_of_range = []
        malformed = []
        
        lbl_files = [
            f for f in lbl_split_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        ]
        
        for lbl_path in lbl_files:
            try:
                lines = lbl_path.read_text(encoding="utf-8").splitlines()
            except Exception as exc:
                logger.error("Could not read labels from %s: %s", lbl_path, exc)
                report["total_errors"] += 1
                continue
                
            for line_no, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                    
                parts = stripped.split()
                report["total_annotations_checked"] += 1
                
                # Check token length
                if len(parts) != 5:
                    err = {
                        "file": lbl_path.name,
                        "line": line_no,
                        "expected_tokens": 5,
                        "got_tokens": len(parts),
                        "raw": stripped
                    }
                    malformed.append(err)
                    report["total_errors"] += 1
                    continue
                    
                # Parse types
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                except ValueError as exc:
                    err = {
                        "file": lbl_path.name,
                        "line": line_no,
                        "raw": stripped,
                        "detail": str(exc)
                    }
                    malformed.append(err)
                    report["total_errors"] += 1
                    continue
                    
                # Validate Class ID
                if cls_id not in valid_class_ids:
                    err = {
                        "file": lbl_path.name,
                        "line": line_no,
                        "class_id": cls_id,
                        "valid_class_ids": sorted(list(valid_class_ids))
                    }
                    invalid_classes.append(err)
                    report["total_errors"] += 1
                else:
                    class_distribution[cls_id] = class_distribution.get(cls_id, 0) + 1
                    
                # Validate Box Coordinates
                coord_errors = []
                if not (0.0 <= cx <= 1.0):
                    coord_errors.append(f"cx={cx:.4f} out of [0,1]")
                if not (0.0 <= cy <= 1.0):
                    coord_errors.append(f"cy={cy:.4f} out of [0,1]")
                if not (0.0 < bw <= 1.0):
                    coord_errors.append(f"bw={bw:.4f} not in (0,1]")
                if not (0.0 < bh <= 1.0):
                    coord_errors.append(f"bh={bh:.4f} not in (0,1]")
                    
                if coord_errors:
                    err = {
                        "file": lbl_path.name,
                        "line": line_no,
                        "issues": coord_errors,
                        "bbox": [cx, cy, bw, bh]
                    }
                    out_of_range.append(err)
                    report["total_errors"] += 1
                    
        if invalid_classes:
            report["invalid_class_ids"][split] = invalid_classes
        if out_of_range:
            report["out_of_range_coords"][split] = out_of_range
        if malformed:
            report["malformed_lines"][split] = malformed
            
    report["class_distribution"] = class_distribution
    return report


def check_cross_split_duplicates(
    images_dir: Path,
    splits: List[str],
    image_extensions: Set[str]
) -> List[Dict[str, Any]]:
    """
    Check if the same image stem is present across multiple splits.
    """
    stem_registry: Dict[str, List[str]] = {}
    for split in splits:
        img_split_dir = images_dir / split
        if not img_split_dir.exists():
            continue
        for img_path in img_split_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                stem_registry.setdefault(img_path.stem, []).append(split)
                
    duplicate_stems = []
    for stem, split_list in stem_registry.items():
        if len(split_list) > 1:
            duplicate_stems.append({
                "stem": stem,
                "found_in_splits": split_list
            })
    return duplicate_stems
