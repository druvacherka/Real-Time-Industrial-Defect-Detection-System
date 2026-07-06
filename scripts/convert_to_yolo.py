#!/usr/bin/env python3
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

RAW_DIR = Path("datasets/raw/NEU-DET")
OUTPUT_DIR = Path("dataset/yolo")

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return (x * dw, y * dh, w * dw, h * dh)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    boxes = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text.strip().lower()
        if class_name not in CLASS_MAP:
            continue
        class_id = CLASS_MAP[class_name]
        
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Convert to YOLO format
        yolo_box = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
        boxes.append((class_id, *yolo_box))
        
    return boxes

def main():
    print("=== Converting XMLs to YOLO and Splitting ===")
    
    # Collect all image-xml pairs
    data_by_class = {c: [] for c in CLASSES}
    
    for split in ["train", "validation"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            continue
        
        img_dir = split_dir / "images"
        xml_dir = split_dir / "annotations"
        
        for class_name in CLASSES:
            class_img_dir = img_dir / class_name
            if not class_img_dir.exists():
                continue
            
            for img_path in class_img_dir.glob("*.jpg"):
                xml_path = xml_dir / f"{img_path.stem}.xml"
                if xml_path.exists():
                    data_by_class[class_name].append((img_path, xml_path))
                    
    # Fixed seed for reproducibility
    random.seed(42)
    
    splits = ["train", "val", "test"]
    for split in splits:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Perform stratified split (70% train, 20% val, 10% test)
    for class_name, pairs in data_by_class.items():
        random.shuffle(pairs)
        total = len(pairs)
        train_idx = int(0.7 * total)
        val_idx = int(0.9 * total)
        
        class_splits = {
            "train": pairs[:train_idx],
            "val": pairs[train_idx:val_idx],
            "test": pairs[val_idx:]
        }
        
        for split, split_pairs in class_splits.items():
            print(f"Split {split} - {class_name}: {len(split_pairs)} files")
            for img_path, xml_path in split_pairs:
                # Copy image
                dst_img = OUTPUT_DIR / "images" / split / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Parse XML and write labels
                boxes = parse_xml(xml_path)
                dst_lbl = OUTPUT_DIR / "labels" / split / f"{img_path.stem}.txt"
                with open(dst_lbl, "w") as f:
                    for box in boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                        
    print("\nDataset split and conversion completed successfully.")

if __name__ == "__main__":
    main()
