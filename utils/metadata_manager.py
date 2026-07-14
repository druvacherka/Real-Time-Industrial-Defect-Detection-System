"""
Metadata Manager Module
=======================
Real-Time Industrial Defect Detection System

Provides structured version tracking and automatic metadata generation
for the NEU Metal Surface Defects dataset (YOLO format).
Saves statistics including image count, class counts, annotation counts,
and storage size to a structured JSON database.

Author: saniyamirjanavar-hash
Date: 2026-07-14
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import os

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("metadata_manager")


class MetadataManager:
    """
    Manages dataset versions, statistics calculation, metadata file generation,
    and validation logging.
    """

    def __init__(self, metadata_path: Optional[Path] = None):
        # Set default path in the dataset directory if not provided
        self.metadata_path = metadata_path or ProjectConfig.ROOT_DIR / "dataset" / "metadata.json"
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata file if it exists, otherwise initialize default structure."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded existing metadata from {self.metadata_path}")
                    return data
            except Exception as e:
                logger.error(f"Error reading metadata file {self.metadata_path}: {e}. Initializing new.")
        
        return {
            "current_version": "v1.0.0",
            "history": []
        }

    def save(self) -> None:
        """Save current metadata dict to disk."""
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved metadata to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to write metadata to {self.metadata_path}: {e}")
            raise

    def get_latest_version(self) -> str:
        """Get the current active dataset version string."""
        return self.metadata.get("current_version", "v1.0.0")

    def increment_version(self, patch: bool = True, minor: bool = False, major: bool = False) -> str:
        """
        Increment version string using semantic versioning format (major.minor.patch).
        """
        current = self.get_latest_version().lstrip("v")
        try:
            parts = [int(p) for p in current.split(".")]
            if len(parts) != 3:
                parts = [1, 0, 0]
        except ValueError:
            parts = [1, 0, 0]

        if major:
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif minor:
            parts[1] += 1
            parts[2] = 0
        elif patch:
            parts[2] += 1

        new_version = f"v{parts[0]}.{parts[1]}.{parts[2]}"
        self.metadata["current_version"] = new_version
        logger.info(f"Incremented dataset version from v{current} to {new_version}")
        return new_version

    def calculate_split_statistics(self, split_dir: Path) -> Dict[str, Any]:
        """
        Scan a specific split directory (e.g. dataset/yolo/images/train)
        and extract image count, file size, class count, and annotations.
        """
        images_path = split_dir
        # Derived labels path
        labels_path = split_dir.parent.parent / "labels" / split_dir.name

        image_count = 0
        annotation_count = 0
        total_size_bytes = 0
        class_distribution: Dict[int, int] = {}

        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        if not images_path.exists():
            logger.warning(f"Images directory {images_path} does not exist.")
            return {
                "image_count": 0,
                "annotation_count": 0,
                "total_size_mb": 0.0,
                "class_distribution": {}
            }

        # Scan images
        for entry in os.scandir(images_path):
            if entry.is_file() and Path(entry.name).suffix.lower() in allowed_extensions:
                image_count += 1
                total_size_bytes += entry.stat().st_size

        # Scan annotations if label directory exists
        if labels_path.exists():
            for entry in os.scandir(labels_path):
                if entry.is_file() and entry.name.endswith(".txt"):
                    file_size = entry.stat().st_size
                    total_size_bytes += file_size
                    try:
                        with open(entry.path, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if parts:
                                    class_id = int(parts[0])
                                    annotation_count += 1
                                    class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                    except Exception as e:
                        logger.warning(f"Could not read label file {entry.path}: {e}")

        return {
            "image_count": image_count,
            "annotation_count": annotation_count,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 3),
            "class_distribution": {str(k): v for k, v in sorted(class_distribution.items())}
        }

    def generate_metadata(self, version: Optional[str] = None, description: str = "") -> Dict[str, Any]:
        """
        Walk all YOLO splits, calculate dataset-wide stats, and save to version history.
        """
        version_str = version or self.get_latest_version()
        yolo_root = ProjectConfig.ROOT_DIR / "dataset" / "yolo"
        
        logger.info(f"Generating automatic dataset metadata statistics for version {version_str}...")
        
        splits_stats: Dict[str, Any] = {}
        total_images = 0
        total_annotations = 0
        total_size_mb = 0.0
        combined_class_dist: Dict[str, int] = {}

        for split in ["train", "val", "test"]:
            split_dir = yolo_root / "images" / split
            stats = self.calculate_split_statistics(split_dir)
            splits_stats[split] = stats
            
            total_images += stats["image_count"]
            total_annotations += stats["annotation_count"]
            total_size_mb += stats["total_size_mb"]
            
            for cid, count in stats["class_distribution"].items():
                combined_class_dist[cid] = combined_class_dist.get(cid, 0) + count

        version_data = {
            "version": version_str,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": description,
            "totals": {
                "image_count": total_images,
                "annotation_count": total_annotations,
                "class_count": len(combined_class_dist),
                "total_size_mb": round(total_size_mb, 3),
                "class_distribution": combined_class_dist
            },
            "splits": splits_stats
        }

        # Check if version exists in history and replace, otherwise append
        history = self.metadata.setdefault("history", [])
        existing_idx = next((i for i, v in enumerate(history) if v["version"] == version_str), -1)
        if existing_idx != -1:
            history[existing_idx] = version_data
            logger.info(f"Updated metadata history for version {version_str}")
        else:
            history.append(version_data)
            logger.info(f"Appended new metadata history record for version {version_str}")

        self.save()
        return version_data


# Reusable utility functions
def generate_current_metadata(description: str = "Automated metadata generation") -> Dict[str, Any]:
    """Helper function to run metadata manager and generate current metadata statistics."""
    manager = MetadataManager()
    return manager.generate_metadata(description=description)
