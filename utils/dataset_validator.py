"""
Dataset Validation Helpers
===========================
Provides validation routines to ensure dataset integrity before
training. Checks for missing files, corrupted images, label
consistency, and class balance.

Author: druvacherka
Date: 2026-07-04
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import cv2
import numpy as np

from utils.constants import CLASS_NAMES, NUM_CLASSES
from utils.data_loader import discover_images, discover_annotations, pair_images_and_annotations
from utils.annotation_handler import parse_voc_annotation, validate_annotation
from utils.logger import get_logger

logger = get_logger(__name__)


def check_image_integrity(image_paths: List[str]) -> Dict[str, List[str]]:
    """
    Verify that every image file can be decoded successfully.

    Args:
        image_paths: List of image file paths.

    Returns:
        Dict with 'valid' and 'corrupted' lists.
    """
    valid: List[str] = []
    corrupted: List[str] = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            corrupted.append(path)
            logger.warning("Corrupted image: %s", path)
        else:
            valid.append(path)

    logger.info(
        "Image integrity: %d valid, %d corrupted out of %d total",
        len(valid), len(corrupted), len(image_paths),
    )
    return {"valid": valid, "corrupted": corrupted}


def check_annotation_integrity(
    annotation_paths: List[str],
) -> Dict[str, List[str]]:
    """
    Verify that every annotation file is well-formed XML.

    Args:
        annotation_paths: List of annotation file paths.

    Returns:
        Dict with 'valid', 'invalid', and 'errors' lists.
    """
    valid: List[str] = []
    invalid: List[str] = []
    errors: List[str] = []

    for path in annotation_paths:
        result = parse_voc_annotation(path)
        if result is None:
            invalid.append(path)
            errors.append(f"Parse failed: {path}")
            continue

        issues = validate_annotation(result)
        if issues:
            invalid.append(path)
            for issue in issues:
                errors.append(f"{path}: {issue}")
        else:
            valid.append(path)

    logger.info(
        "Annotation integrity: %d valid, %d invalid out of %d total",
        len(valid), len(invalid), len(annotation_paths),
    )
    return {"valid": valid, "invalid": invalid, "errors": errors}


def compute_class_distribution(
    annotation_paths: List[str],
) -> Dict[str, int]:
    """
    Count object instances per class across all annotations.

    Args:
        annotation_paths: List of annotation file paths.

    Returns:
        Dict mapping class name to count.
    """
    counter: Counter = Counter()

    for path in annotation_paths:
        result = parse_voc_annotation(path)
        if result is None:
            continue
        for obj in result.get("objects", []):
            counter[obj["name"]] += 1

    distribution = {name: counter.get(name, 0) for name in CLASS_NAMES}
    logger.info("Class distribution: %s", distribution)
    return distribution


def find_duplicate_images(image_paths: List[str]) -> List[List[str]]:
    """
    Detect duplicate images by comparing file sizes and checksums.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of groups, where each group contains paths of identical images.
    """
    import hashlib

    hash_map: Dict[str, List[str]] = {}

    for path in image_paths:
        try:
            with open(path, "rb") as fh:
                file_hash = hashlib.md5(fh.read()).hexdigest()
            hash_map.setdefault(file_hash, []).append(path)
        except OSError:
            logger.warning("Cannot read file for hashing: %s", path)

    duplicates = [group for group in hash_map.values() if len(group) > 1]
    if duplicates:
        logger.warning("Found %d groups of duplicate images", len(duplicates))
    else:
        logger.info("No duplicate images found")
    return duplicates


def validate_dataset(
    image_dir: str,
    annotation_dir: str,
) -> Dict[str, any]:
    """
    Run a complete validation suite on the dataset.

    Args:
        image_dir: Directory containing images.
        annotation_dir: Directory containing annotations.

    Returns:
        Comprehensive validation report dictionary.
    """
    logger.info("Starting dataset validation...")

    images = discover_images(image_dir)
    annotations = discover_annotations(annotation_dir)
    pairs = pair_images_and_annotations(image_dir, annotation_dir)

    img_check = check_image_integrity(images)
    ann_check = check_annotation_integrity(annotations)
    class_dist = compute_class_distribution(annotations)
    duplicates = find_duplicate_images(images)

    unpaired_images = len(images) - len(pairs)
    unpaired_anns = len(annotations) - len(pairs)

    report = {
        "total_images": len(images),
        "total_annotations": len(annotations),
        "paired": len(pairs),
        "unpaired_images": unpaired_images,
        "unpaired_annotations": unpaired_anns,
        "corrupted_images": len(img_check["corrupted"]),
        "invalid_annotations": len(ann_check["invalid"]),
        "duplicate_groups": len(duplicates),
        "class_distribution": class_dist,
        "annotation_errors": ann_check["errors"],
        "is_valid": (
            len(img_check["corrupted"]) == 0
            and len(ann_check["invalid"]) == 0
            and unpaired_images == 0
        ),
    }

    status = "PASS" if report["is_valid"] else "ISSUES FOUND"
    logger.info("Dataset validation complete — %s", status)
    return report
