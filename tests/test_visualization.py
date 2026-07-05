"""
Unit tests for visualization utility modules.
"""

import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from utils.visualization import parse_voc_xml, draw_annotations


class TestVisualization(unittest.TestCase):
    
    def setUp(self):
        # Create a temp directory for mock xml and images
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_dir_path = Path(self.test_dir.name)
        
        # Write a mock Pascal VOC XML file
        self.mock_xml_content = """<annotation>
            <folder>images</folder>
            <filename>mock_image.jpg</filename>
            <size>
                <width>200</width>
                <height>200</height>
                <depth>3</depth>
            </size>
            <object>
                <name>scratches</name>
                <bndbox>
                    <xmin>15</xmin>
                    <ymin>30</ymin>
                    <xmax>150</xmax>
                    <ymax>180</ymax>
                </bndbox>
            </object>
            <object>
                <name>crazing</name>
                <bndbox>
                    <xmin>5</xmin>
                    <ymin>10</ymin>
                    <xmax>50</xmax>
                    <ymax>60</ymax>
                </bndbox>
            </object>
        </annotation>"""
        
        self.xml_path = self.test_dir_path / "mock_image.xml"
        with open(self.xml_path, "w") as f:
            f.write(self.mock_xml_content)
            
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_parse_voc_xml(self):
        """Tests that XML annotation file is parsed correctly."""
        size, objects = parse_voc_xml(self.xml_path)
        
        # Verify size
        self.assertEqual(size, (200, 200))
        
        # Verify number of objects parsed
        self.assertEqual(len(objects), 2)
        
        # Verify object 1
        self.assertEqual(objects[0]["class"], "scratches")
        self.assertEqual(objects[0]["bbox"], (15, 30, 150, 180))
        
        # Verify object 2
        self.assertEqual(objects[1]["class"], "crazing")
        self.assertEqual(objects[1]["bbox"], (5, 10, 50, 60))
        
    def test_draw_annotations(self):
        """Tests drawing bounding box annotations on a dummy image."""
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, objects = parse_voc_xml(self.xml_path)
        
        annotated = draw_annotations(dummy_img, objects)
        
        # Check output is a valid numpy array with same dimensions
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, dummy_img.shape)
        
        # Check that drawn pixels are modified (image is no longer all zeros)
        self.assertTrue(np.any(annotated > 0))


if __name__ == "__main__":
    unittest.main()
