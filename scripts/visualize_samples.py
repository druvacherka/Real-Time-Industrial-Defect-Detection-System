#!/usr/bin/env python3
"""
Sample Visualization Script for Real-Time Industrial Defect Detection System.
Randomly selects images from the dataset and displays them with bounding box overlays.
"""

import random
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import yaml

# Paths definitions
DATASET_ROOT = Path("dataset/yolo")
CLASSES_CFG = Path("configs/classes.yaml")


def load_class_names():
    """Load class mapping from configuration file."""
    if CLASSES_CFG.exists():
        try:
            with open(CLASSES_CFG, "r") as f:
                data = yaml.safe_load(f)
                return data.get("classes", [])
        except Exception as e:
            print(f"Warning: Failed to load classes config: {e}")
    # Fallback default NEU class names
    return ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]


def draw_bounding_boxes(image, label_path, class_names):
    """Draws YOLO normalized bounding boxes on the image."""
    h, w, _ = image.shape
    # Unique colors for each class (up to 6)
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                x_center, y_center, bbox_w, bbox_h = map(float, parts[1:5])

                # Denormalize coordinates
                x1 = int((x_center - bbox_w / 2) * w)
                y1 = int((y_center - bbox_h / 2) * h)
                x2 = int((x_center + bbox_w / 2) * w)
                y2 = int((y_center + bbox_h / 2) * h)

                # Get class label and color
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                color = colors[class_id % len(colors)]

                # Draw rectangle and label text
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, class_name, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")

    return image


def visualize_random_samples(num_samples=4):
    """Randomly selects and visualizes images from the YOLO dataset splits."""
    print("=" * 60)
    print("DATASET SAMPLE VISUALIZATION")
    print("=" * 60)

    class_names = load_class_names()
    splits = ["train", "val", "test"]

    # Gather all image files
    all_images = []
    for split in splits:
        img_dir = DATASET_ROOT / "images" / split
        if img_dir.exists():
            images = [f for f in img_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            for img in images:
                all_images.append((split, img))

    if not all_images:
        print("No images found in the dataset folder structure.")
        print(f"Please place image files in: {DATASET_ROOT}/images/[train/val/test]")
        return

    # Select random samples
    samples_to_show = random.sample(all_images, min(len(all_images), num_samples))
    fig, axes = plt.subplots(1, len(samples_to_show), figsize=(15, 5))
    if len(samples_to_show) == 1:
        axes = [axes]

    for idx, (split, img_path) in enumerate(samples_to_show):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Check for corresponding label
        label_filename = img_path.stem + ".txt"
        label_path = DATASET_ROOT / "labels" / split / label_filename

        if label_path.exists():
            image = draw_bounding_boxes(image, label_path, class_names)
        else:
            print(f"No label found for {img_path.name}")

        # Convert BGR (OpenCV) to RGB (Matplotlib)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(image_rgb)
        axes[idx].set_title(f"[{split.upper()}] {img_path.name}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_random_samples()
