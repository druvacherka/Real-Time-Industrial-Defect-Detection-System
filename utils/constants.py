"""
Global constants used across the project.
"""

PROJECT_NAME = "Real-Time Industrial Defect Detection System"

MODEL_NAME = "YOLOv8"

DATASET_NAME = "NEU-DET"

SUPPORTED_IMAGE_FORMATS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]

SUPPORTED_ANNOTATION_FORMAT = ".xml"

NUM_CLASSES = 6

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]