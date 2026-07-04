"""
Tests for Data Loader and Annotation Utilities
================================================
Unit tests for the data loading, annotation parsing,
and dataset validation modules.

Author: druvacherka
Date: 2026-07-04
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from utils.data_loader import (
    discover_images,
    discover_annotations,
    load_image,
    pair_images_and_annotations,
    get_class_index,
    compute_image_statistics,
)
from utils.annotation_handler import (
    parse_voc_annotation,
    voc_to_yolo,
    voc_to_coco,
    validate_annotation,
)
from utils.dataset_validator import (
    check_image_integrity,
    compute_class_distribution,
    find_duplicate_images,
)


class TestDataLoader:
    """Tests for data_loader module."""

    def test_discover_images_empty_dir(self, tmp_path):
        """Should return empty list for directory with no images."""
        result = discover_images(str(tmp_path))
        assert result == []

    def test_discover_images_finds_jpg(self, tmp_path):
        """Should find .jpg files."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        result = discover_images(str(tmp_path))
        assert len(result) == 1

    def test_load_image_valid(self, tmp_path):
        """Should load a valid image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        loaded = load_image(str(path))
        assert loaded is not None
        assert loaded.shape == (100, 100, 3)

    def test_load_image_missing(self):
        """Should return None for missing file."""
        result = load_image("/nonexistent/path.jpg")
        assert result is None

    def test_load_image_resize(self, tmp_path):
        """Should resize image to target size."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), img)
        loaded = load_image(str(path), target_size=(100, 100))
        assert loaded.shape == (100, 100, 3)

    def test_get_class_index_valid(self):
        """Should return correct index for known classes."""
        assert get_class_index("crazing") == 0
        assert get_class_index("scratches") == 5

    def test_get_class_index_invalid(self):
        """Should raise ValueError for unknown class."""
        with pytest.raises(ValueError):
            get_class_index("unknown_class")

    def test_compute_image_statistics(self):
        """Should compute per-channel statistics."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        stats = compute_image_statistics(img)
        assert stats["blue_mean"] == 128.0
        assert stats["green_mean"] == 128.0
        assert stats["red_mean"] == 128.0


class TestAnnotationHandler:
    """Tests for annotation_handler module."""

    def _create_voc_xml(self, tmp_path, filename="test.jpg"):
        """Helper to create a minimal VOC XML file."""
        xml_content = f"""<?xml version="1.0"?>
<annotation>
    <filename>{filename}</filename>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>
    <object>
        <name>crazing</name>
        <difficult>0</difficult>
        <bndbox>
            <xmin>10</xmin>
            <ymin>20</ymin>
            <xmax>100</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>"""
        xml_path = tmp_path / "test.xml"
        xml_path.write_text(xml_content)
        return str(xml_path)

    def test_parse_valid_annotation(self, tmp_path):
        """Should parse a valid VOC XML correctly."""
        xml_path = self._create_voc_xml(tmp_path)
        result = parse_voc_annotation(xml_path)
        assert result is not None
        assert result["filename"] == "test.jpg"
        assert result["size"]["width"] == 200
        assert len(result["objects"]) == 1
        assert result["objects"][0]["name"] == "crazing"

    def test_parse_missing_file(self):
        """Should return None for missing file."""
        result = parse_voc_annotation("/nonexistent/file.xml")
        assert result is None

    def test_voc_to_yolo_conversion(self):
        """Should convert VOC bbox to normalized YOLO format."""
        bbox = [10.0, 20.0, 100.0, 150.0]
        x_c, y_c, w, h = voc_to_yolo(bbox, 200, 200)
        assert abs(x_c - 0.275) < 1e-6
        assert abs(y_c - 0.425) < 1e-6
        assert abs(w - 0.45) < 1e-6
        assert abs(h - 0.65) < 1e-6

    def test_voc_to_coco_conversion(self):
        """Should convert VOC bbox to COCO format."""
        bbox = [10.0, 20.0, 100.0, 150.0]
        x, y, w, h = voc_to_coco(bbox)
        assert x == 10.0
        assert y == 20.0
        assert w == 90.0
        assert h == 130.0

    def test_validate_annotation_valid(self, tmp_path):
        """Should return empty list for valid annotation."""
        xml_path = self._create_voc_xml(tmp_path)
        result = parse_voc_annotation(xml_path)
        errors = validate_annotation(result)
        assert errors == []


class TestDatasetValidator:
    """Tests for dataset_validator module."""

    def test_check_image_integrity(self, tmp_path):
        """Should classify valid and corrupted images."""
        valid_img = np.zeros((50, 50, 3), dtype=np.uint8)
        valid_path = tmp_path / "valid.jpg"
        cv2.imwrite(str(valid_path), valid_img)

        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not an image")

        result = check_image_integrity([str(valid_path), str(corrupt_path)])
        assert len(result["valid"]) == 1
        assert len(result["corrupted"]) == 1

    def test_find_duplicate_images(self, tmp_path):
        """Should detect duplicate image files."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path1 = tmp_path / "img1.jpg"
        path2 = tmp_path / "img2.jpg"
        cv2.imwrite(str(path1), img)
        cv2.imwrite(str(path2), img)
        duplicates = find_duplicate_images([str(path1), str(path2)])
        assert len(duplicates) == 1
        assert len(duplicates[0]) == 2
