"""
Script to visualize dataset annotations by drawing bounding boxes.
"""

import argparse
import os
import random
import sys
from pathlib import Path
import cv2

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.visualization import parse_voc_xml, draw_annotations


def main():
    parser = argparse.ArgumentParser(description="Visualize raw dataset annotations.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to visualize from (train or validation)."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to visualize."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ProjectConfig.ROOT_DIR / "results" / "visualizations"),
        help="Directory to save visualized images."
    )
    
    args = parser.parse_args()
    
    split_dir = ProjectConfig.RAW_DATASET / "NEU-DET" / args.split
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    
    if not images_dir.exists() or not annotations_dir.exists():
        print(f"Error: Dataset split directory {split_dir} does not contain images/ or annotations/")
        sys.exit(1)
        
    image_files = list(images_dir.rglob("*.jpg"))
    if not image_files:
        print(f"Error: No JPEG images found in {images_dir}")
        sys.exit(1)
        
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    samples = random.sample(image_files, min(args.num_samples, len(image_files)))
    print(f"Visualizing {len(samples)} samples from {args.split} split...")
    
    for img_path in samples:
        xml_path = annotations_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            print(f"Warning: Annotation XML file not found for {img_path.name}")
            continue
            
        print(f"Processing: {img_path.name} | XML: {xml_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
            
        # Parse XML
        _, objects = parse_voc_xml(xml_path)
        
        # Draw annotations
        annotated_image = draw_annotations(image, objects)
        
        # Save output
        save_path = output_path / f"visualized_{img_path.name}"
        cv2.imwrite(str(save_path), annotated_image)
        print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()
