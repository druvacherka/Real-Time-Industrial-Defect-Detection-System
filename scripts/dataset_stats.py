#!/usr/bin/env python3
"""
Dataset Statistics and Report Generator (Refactored)
===================================================
Real-Time Industrial Defect Detection System

Main script to compute dataset metrics and generate visualizations.
Utilizes the modular library in utils.dataset_statistics.
Generates reports/dataset_statistics_report.md.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Bootstrap path resolution
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.config import (
    IMAGES_DIR,
    LABELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    CLASS_NAMES,
    SPLITS,
    CLASS_COLORS_BGR,
    ensure_dirs,
)
from utils.dataset_statistics import (
    calculate_stats,
    generate_visualizations,
    write_statistics_report,
)

# Initialize logging
ensure_dirs(LOGS_DIR, REPORTS_DIR)
logger = logging.getLogger("dataset_statistics")
logger.setLevel(logging.INFO)

if logger.handlers:
    logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# File handler
fh = logging.FileHandler(LOGS_DIR / "dataset_statistics.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def main() -> None:
    logger.info("Starting dataset statistics calculation pipeline...")

    # Calculate statistics
    stats = calculate_stats(IMAGES_DIR, LABELS_DIR, SPLITS, CLASS_NAMES)

    # Output directories
    vis_dir = REPORTS_DIR / "visualizations"
    report_path = REPORTS_DIR / "dataset_statistics_report.md"

    # Generate charts and drawings
    generate_visualizations(
        stats,
        vis_dir,
        CLASS_NAMES,
        IMAGES_DIR,
        LABELS_DIR,
        SPLITS,
        CLASS_COLORS_BGR
    )

    # Write MD report
    write_statistics_report(report_path, stats, CLASS_NAMES)

    # Save stats to json for downstream use
    stats_json_path = REPORTS_DIR / "dataset_statistics.json"
    import json
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Dataset statistics pipeline completed successfully.")


if __name__ == "__main__":
    main()
