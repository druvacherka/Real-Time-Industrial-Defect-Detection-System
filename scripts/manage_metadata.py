#!/usr/bin/env python3
"""
Manage Metadata script
======================
Real-Time Industrial Defect Detection System

Invokes the MetadataManager to calculate dataset totals (images, annotations, classes, sizes),
manages semantic version numbers, and outputs database records.

Author: saniyamirjanavar-hash
Date: 2026-07-14
"""

import sys
from pathlib import Path

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.metadata_manager import MetadataManager


def main() -> None:
    print("Initializing Metadata Manager...")
    manager = MetadataManager()
    
    # Generate metadata for current version
    version = manager.get_latest_version()
    print(f"Generating stats for current version: {version}")
    
    stats = manager.generate_metadata(description="Initial automated dataset versioning setup")
    
    print("\n--- DATASET METADATA SUMMARY ---")
    print(f"Version:            {stats['version']}")
    print(f"Timestamp:          {stats['timestamp']}")
    print(f"Total Images:       {stats['totals']['image_count']}")
    print(f"Total Annotations:  {stats['totals']['annotation_count']}")
    print(f"Total Size:         {stats['totals']['total_size_mb']} MB")
    print(f"Classes Found:      {stats['totals']['class_count']}")
    print("Class Distribution:")
    for cid, count in stats['totals']['class_distribution'].items():
        print(f"  Class {cid}: {count}")
    print("--------------------------------\n")
    print("Metadata generation completed successfully.")


if __name__ == "__main__":
    main()
