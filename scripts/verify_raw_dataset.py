#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

RAW_DIR = Path("datasets/raw/NEU-DET")

def get_image_and_xml_pairs():
    splits = ["train", "validation"]
    image_paths = {}
    xml_paths = {}
    
    for split in splits:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            continue
        
        # XMLs
        xml_dir = split_dir / "annotations"
        if xml_dir.exists():
            for xml_file in xml_dir.glob("*.xml"):
                xml_paths[xml_file.stem] = xml_file
                
        # Images
        img_dir = split_dir / "images"
        if img_dir.exists():
            for class_dir in img_dir.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.glob("*.jpg"):
                        image_paths[img_file.stem] = img_file
                        
    return image_paths, xml_paths

def verify():
    print("=== Raw Dataset Verification ===")
    image_paths, xml_paths = get_image_and_xml_pairs()
    
    print(f"Total image files found: {len(image_paths)}")
    print(f"Total XML files found: {len(xml_paths)}")
    
    missing_xml = []
    missing_img = []
    corrupted = []
    duplicates = {}
    hashes = {}
    
    # Check for missing XMLs
    for stem, img_path in image_paths.items():
        if stem not in xml_paths:
            missing_xml.append(img_path)
            
    # Check for missing images
    for stem, xml_path in xml_paths.items():
        if stem not in image_paths:
            missing_img.append(xml_path)
            
    # Check integrity & duplicates
    for stem, img_path in image_paths.items():
        # Check corrupt
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                corrupted.append(img_path)
        except Exception:
            corrupted.append(img_path)
            continue
            
        # Check duplicate
        try:
            with open(img_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            if h in hashes:
                duplicates.setdefault(h, [hashes[h]]).append(img_path)
            else:
                hashes[h] = img_path
        except Exception:
            pass
            
    print("\n--- Verification Results ---")
    print(f"Missing XML annotations: {len(missing_xml)}")
    for p in missing_xml[:5]:
        print(f"  - {p}")
        
    print(f"Missing Image files: {len(missing_img)}")
    for p in missing_img[:5]:
        print(f"  - {p}")
        
    print(f"Corrupted Images: {len(corrupted)}")
    for p in corrupted[:5]:
        print(f"  - {p}")
        
    print(f"Duplicate image sets found: {len(duplicates)}")
    for h, paths in list(duplicates.items())[:5]:
        print(f"  - Hash {h}: {[p.name for p in paths]}")
        
    success = (len(missing_xml) == 0 and len(missing_img) == 0 and len(corrupted) == 0)
    if success:
        print("\nVerification PASSED: All images match annotations and have no corruption.")
    else:
        print("\nVerification FAILED: Issues found.")
        
    return success

if __name__ == "__main__":
    verify()
