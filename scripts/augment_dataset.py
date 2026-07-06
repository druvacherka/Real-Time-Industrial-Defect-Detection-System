#!/usr/bin/env python3
import os
import random
from pathlib import Path
import cv2
import albumentations as A
import numpy as np

# Paths
YOLO_DIR = Path("dataset/yolo")
TRAIN_IMG_DIR = YOLO_DIR / "images" / "train"
TRAIN_LBL_DIR = YOLO_DIR / "labels" / "train"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 1. Define Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.MotionBlur(p=0.3),
    A.HueSaturationValue(p=0.5),
    A.CLAHE(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))

def load_yolo_labels(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                # Ensure coordinates are within [0, 1]
                coords = [max(0.0, min(1.0, c)) for c in coords]
                # Albumentations requirements: w > 0, h > 0
                if coords[2] > 0 and coords[3] > 0:
                    boxes.append([class_id] + coords)
    return boxes

def save_yolo_labels(label_path, bboxes):
    with open(label_path, "w") as f:
        for box in bboxes:
            class_id, x, y, w, h = box
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def main():
    print("=== Albumentations Augmentation & Class Balancing Pipeline ===")
    
    # Analyze current training class distributions (instances/objects)
    image_files = list(TRAIN_IMG_DIR.glob("*.jpg"))
    class_counts = {i: 0 for i in range(len(CLASSES))}
    image_mapped = {i: [] for i in range(len(CLASSES))}
    
    for img_path in image_files:
        lbl_path = TRAIN_LBL_DIR / f"{img_path.stem}.txt"
        boxes = load_yolo_labels(lbl_path)
        
        # Keep track of which images contain which classes
        classes_in_img = set()
        for box in boxes:
            class_id = box[0]
            class_counts[class_id] += 1
            classes_in_img.add(class_id)
            
        for cid in classes_in_img:
            image_mapped[cid].append(img_path)
            
    print("Initial training instance counts:")
    for cid, count in class_counts.items():
        print(f"  {CLASSES[cid]}: {count}")
        
    target_count = max(class_counts.values())
    print(f"Target instance count for balancing: {target_count}")
    
    # Perform augmentation for minority classes
    for cid, count in class_counts.items():
        if count >= target_count:
            continue
            
        class_name = CLASSES[cid]
        print(f"\nAugmenting minority class: {class_name} ({count} -> {target_count})")
        
        candidates = image_mapped[cid]
        if not candidates:
            print(f"No candidate images found for class {class_name}. Skipping.")
            continue
            
        aug_idx = 0
        current_count = count
        
        # Loop until target count is met
        while current_count < target_count:
            # Pick a random candidate image
            src_img_path = random.choice(candidates)
            src_lbl_path = TRAIN_LBL_DIR / f"{src_img_path.stem}.txt"
            
            image = cv2.imread(str(src_img_path))
            if image is None:
                continue
                
            yolo_boxes = load_yolo_labels(src_lbl_path)
            if not yolo_boxes:
                continue
                
            # Prepare bounding boxes for Albumentations
            bboxes = [box[1:] for box in yolo_boxes]
            class_labels = [box[0] for box in yolo_boxes]
            
            # Apply transform
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
            except Exception as e:
                # Bounding box coordinates occasionally edge out of [0, 1] during rotate, skip
                continue
                
            if not aug_bboxes:
                continue
                
            # Save augmented image and labels
            dst_stem = f"aug_{src_img_path.stem}_{aug_idx}"
            dst_img_path = TRAIN_IMG_DIR / f"{dst_stem}.jpg"
            dst_lbl_path = TRAIN_LBL_DIR / f"{dst_stem}.txt"
            
            cv2.imwrite(str(dst_img_path), aug_img)
            
            # Re-assemble boxes: [class_id, x, y, w, h]
            assembled_boxes = []
            for lbl, bbox in zip(aug_labels, aug_bboxes):
                assembled_boxes.append([lbl, *bbox])
                if lbl == cid:
                    current_count += 1
                    
            save_yolo_labels(dst_lbl_path, assembled_boxes)
            aug_idx += 1
            
        print(f"Generated {aug_idx} augmented images for {class_name}. Final count: {current_count}")
        
    print("\nAugmentation and class balancing completed.")

if __name__ == "__main__":
    main()
