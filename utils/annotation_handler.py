"""
Annotation Format Handler
=========================
Parses Pascal VOC XML annotations used in the NEU Metal Surface
Defects dataset and converts bounding boxes to different formats.

Author: druvacherka
Date: 2026-07-04
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


def parse_voc_annotation(xml_path: str) -> Optional[Dict]:
    """
    Parse a Pascal VOC XML annotation file.

    Args:
        xml_path: Path to the XML annotation file.

    Returns:
        Dict with keys: filename, size (width, height, depth),
        and objects (list of dicts with name, bbox).
        Returns None if parsing fails.
    """
    if not Path(xml_path).is_file():
        logger.error("Annotation file not found: %s", xml_path)
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.error("XML parse error in %s: %s", xml_path, exc)
        return None

    filename_elem = root.find("filename")
    filename = filename_elem.text if filename_elem is not None else ""

    size_elem = root.find("size")
    width = int(size_elem.findtext("width", "0"))
    height = int(size_elem.findtext("height", "0"))
    depth = int(size_elem.findtext("depth", "1"))

    objects: List[Dict] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "unknown")
        difficult = int(obj.findtext("difficult", "0"))

        bndbox = obj.find("bndbox")
        if bndbox is None:
            logger.warning("Missing bndbox in %s", xml_path)
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        objects.append(
            {
                "name": name,
                "difficult": difficult,
                "bbox": [xmin, ymin, xmax, ymax],
            }
        )

    return {
        "filename": filename,
        "size": {"width": width, "height": height, "depth": depth},
        "objects": objects,
    }


def voc_to_yolo(
    bbox: List[float], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert a Pascal VOC bounding box to YOLO format.

    Args:
        bbox: [xmin, ymin, xmax, ymax] in absolute pixels.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1].
    """
    xmin, ymin, xmax, ymax = bbox
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    return (x_center, y_center, w, h)


def voc_to_coco(
    bbox: List[float],
) -> Tuple[float, float, float, float]:
    """
    Convert a Pascal VOC bounding box to COCO format.

    Args:
        bbox: [xmin, ymin, xmax, ymax].

    Returns:
        (x, y, width, height) — top-left corner + size.
    """
    xmin, ymin, xmax, ymax = bbox
    return (xmin, ymin, xmax - xmin, ymax - ymin)


def validate_annotation(annotation: Dict) -> List[str]:
    """
    Validate a parsed annotation dictionary.

    Args:
        annotation: Parsed annotation from parse_voc_annotation.

    Returns:
        List of validation error messages. Empty means valid.
    """
    errors: List[str] = []

    if not annotation.get("filename"):
        errors.append("Missing filename")

    size = annotation.get("size", {})
    if size.get("width", 0) <= 0 or size.get("height", 0) <= 0:
        errors.append("Invalid image dimensions")

    for idx, obj in enumerate(annotation.get("objects", [])):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        xmin, ymin, xmax, ymax = bbox
        if xmax <= xmin or ymax <= ymin:
            errors.append(f"Object {idx}: invalid bounding box {bbox}")
        if not obj.get("name"):
            errors.append(f"Object {idx}: missing class name")

    return errors
