"""
Visualization utilities for defect detection dataset.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2
import numpy as np

from utils.constants import CLASS_NAMES


def parse_voc_xml(xml_path: Union[str, Path]) -> Tuple[Tuple[int, int], List[Dict[str, Union[str, Tuple[int, int, int, int]]]]]:
    """
    Parses a Pascal VOC format XML annotation file.
    
    Args:
        xml_path: Path to the XML file.
        
    Returns:
        A tuple containing:
            - (width, height) of the image.
            - List of objects, where each object is a dictionary:
              {"class": str, "bbox": (xmin, ymin, xmax, ymax)}
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {xml_path}")
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size_elem = root.find("size")
    if size_elem is not None:
        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)
    else:
        width, height = 200, 200  # Default fallback for NEU-DET
        
    objects = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))
        
        objects.append({
            "class": class_name,
            "bbox": (xmin, ymin, xmax, ymax)
        })
        
    return (width, height), objects


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """
    Returns a distinct BGR color for each class.
    """
    colors = {
        "crazing": (0, 0, 255),          # Red
        "inclusion": (0, 255, 0),        # Green
        "patches": (255, 0, 0),          # Blue
        "pitted_surface": (0, 255, 255),  # Yellow
        "rolled-in_scale": (255, 0, 255),# Magenta
        "scratches": (255, 255, 0),      # Cyan
    }
    return colors.get(class_name, (255, 255, 255))


def draw_annotations(
    image: np.ndarray,
    objects: List[Dict[str, Union[str, Tuple[int, int, int, int]]]],
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    Draws bounding boxes and labels on the image.
    
    Args:
        image: Source image in BGR format.
        objects: List of dictionaries containing "class" and "bbox".
        line_thickness: Box line thickness.
        font_scale: Font scale for label text.
        
    Returns:
        The annotated image copy.
    """
    annotated_img = image.copy()
    
    for obj in objects:
        class_name = obj["class"]
        xmin, ymin, xmax, ymax = obj["bbox"]
        
        color = get_class_color(class_name)
        
        # Draw box
        cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), color, line_thickness)
        
        # Prepare text label
        label = f"{class_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Put background rectangle for text
        cv2.rectangle(
            annotated_img,
            (xmin, ymin - text_height - 4),
            (xmin + text_width, ymin),
            color,
            -1
        )
        
        # Draw text label (white text on class-colored background)
        cv2.putText(
            annotated_img,
            label,
            (xmin, ymin - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA
        )
        
    return annotated_img
