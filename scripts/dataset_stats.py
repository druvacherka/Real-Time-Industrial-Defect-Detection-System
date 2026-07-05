#!/usr/bin/env python3
"""
Class Distribution Analysis and Dataset Balance Report
========================================================
Real-Time Industrial Defect Detection System

Analyzes the distribution of all six NEU defect classes, generates
statistics, creates Matplotlib visualizations, and exports reports.

Author: saniyamirjanavar-hash
Date: 2026-07-05
"""

import sys
import logging
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving charts
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "class_distribution.log", mode="a"),
    ],
)
logger = logging.getLogger("class_distribution")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "yolo"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
REPORTS_DIR = PROJECT_ROOT / "reports"
GRAPHS_DIR = REPORTS_DIR / "graphs"

SPLITS = ["train", "val", "test"]
DEFECT_CLASSES = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

# Color palette for charts (one per class)
CLASS_COLORS = [
    "#FF6B6B",  # crazing — coral red
    "#4ECDC4",  # inclusion — teal
    "#45B7D1",  # patches — sky blue
    "#96CEB4",  # pitted_surface — sage green
    "#FFEAA7",  # rolled-in_scale — yellow
    "#DDA0DD",  # scratches — plum
]


class ClassDistributionAnalyzer:
    """Analyze defect class distribution across dataset splits."""

    def __init__(self):
        self.stats = {
            "timestamp": datetime.now().isoformat(),
            "splits": {},
            "overall": {},
        }
        # Per-split, per-class annotation counts
        self.split_class_counts = defaultdict(lambda: defaultdict(int))
        # Per-split, per-class image counts (images containing at least one bbox of that class)
        self.split_class_images = defaultdict(lambda: defaultdict(set))
        # Missing labels and empty annotations
        self.missing_labels = defaultdict(list)
        self.empty_annotations = defaultdict(list)

    # ------------------------------------------------------------------
    # Scan labels
    # ------------------------------------------------------------------
    def scan_labels(self):
        """Parse all label files and aggregate class counts."""
        logger.info("Scanning label files across all splits...")

        for split in SPLITS:
            img_dir = IMAGES_DIR / split
            lbl_dir = LABELS_DIR / split

            if not lbl_dir.exists():
                logger.warning(f"  [{split}] Labels directory missing")
                continue

            # Collect image stems
            img_stems = set()
            if img_dir.exists():
                img_stems = {
                    f.stem for f in img_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                }

            lbl_files = sorted([
                f for f in lbl_dir.iterdir()
                if f.is_file() and f.suffix == ".txt"
            ])

            # Check for images missing labels
            lbl_stems = {f.stem for f in lbl_files}
            for stem in img_stems - lbl_stems:
                self.missing_labels[split].append(stem)

            for lbl_path in lbl_files:
                try:
                    with open(lbl_path, "r") as fh:
                        lines = [l.strip() for l in fh.readlines() if l.strip()]

                    if len(lines) == 0:
                        self.empty_annotations[split].append(lbl_path.name)
                        continue

                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            self.split_class_counts[split][cls_id] += 1
                            self.split_class_images[split][cls_id].add(lbl_path.stem)
                except Exception as exc:
                    logger.warning(f"  Could not parse {lbl_path.name}: {exc}")

            total_annotations = sum(self.split_class_counts[split].values())
            logger.info(
                f"  [{split}] Labels: {len(lbl_files)} | "
                f"Annotations: {total_annotations} | "
                f"Missing labels: {len(self.missing_labels[split])} | "
                f"Empty files: {len(self.empty_annotations[split])}"
            )

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    def compute_statistics(self) -> dict:
        """Compute per-class and overall statistics."""
        logger.info("Computing class distribution statistics...")

        overall_counts = defaultdict(int)
        overall_images = defaultdict(set)

        for split in SPLITS:
            split_stats = {}
            for cls_id, cls_name in DEFECT_CLASSES.items():
                count = self.split_class_counts[split].get(cls_id, 0)
                img_count = len(self.split_class_images[split].get(cls_id, set()))
                split_stats[cls_name] = {
                    "annotation_count": count,
                    "image_count": img_count,
                }
                overall_counts[cls_id] += count
                overall_images[cls_id].update(
                    self.split_class_images[split].get(cls_id, set())
                )

            self.stats["splits"][split] = {
                "classes": split_stats,
                "missing_labels": len(self.missing_labels[split]),
                "empty_annotations": len(self.empty_annotations[split]),
            }

        # Overall
        total = sum(overall_counts.values())
        overall_stats = {}
        for cls_id, cls_name in DEFECT_CLASSES.items():
            count = overall_counts[cls_id]
            pct = (count / total * 100) if total > 0 else 0.0
            overall_stats[cls_name] = {
                "annotation_count": count,
                "image_count": len(overall_images[cls_id]),
                "percentage": round(pct, 2),
            }

        self.stats["overall"] = {
            "total_annotations": total,
            "classes": overall_stats,
        }

        # Log summary
        for cls_name, info in overall_stats.items():
            logger.info(
                f"  {cls_name}: {info['annotation_count']} annotations "
                f"({info['percentage']}%) in {info['image_count']} images"
            )

        return self.stats

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def generate_charts(self):
        """Create and save Matplotlib charts."""
        if not HAS_MPL:
            logger.warning("Matplotlib not available — skipping chart generation")
            return

        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        overall = self.stats.get("overall", {}).get("classes", {})
        class_names = list(overall.keys())
        counts = [overall[c]["annotation_count"] for c in class_names]
        percentages = [overall[c]["percentage"] for c in class_names]

        # ---- Bar Chart: Annotation counts per class ----
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(class_names, counts, color=CLASS_COLORS[:len(class_names)], edgecolor="white", linewidth=0.8)
        ax.set_title("Defect Class Distribution — Annotation Counts", fontsize=14, fontweight="bold")
        ax.set_xlabel("Defect Class", fontsize=12)
        ax.set_ylabel("Number of Annotations", fontsize=12)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        bar_path = GRAPHS_DIR / "class_distribution_bar.png"
        fig.savefig(str(bar_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Bar chart saved: {bar_path}")

        # ---- Pie Chart: Class percentage ----
        fig, ax = plt.subplots(figsize=(8, 8))
        if sum(counts) > 0:
            wedges, texts, autotexts = ax.pie(
                counts, labels=class_names, autopct="%1.1f%%",
                colors=CLASS_COLORS[:len(class_names)],
                startangle=140, textprops={"fontsize": 10},
            )
            for autotext in autotexts:
                autotext.set_fontweight("bold")
        else:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                    fontsize=14, transform=ax.transAxes)
        ax.set_title("Defect Class Distribution — Percentage", fontsize=14, fontweight="bold")
        plt.tight_layout()
        pie_path = GRAPHS_DIR / "class_distribution_pie.png"
        fig.savefig(str(pie_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Pie chart saved: {pie_path}")

        # ---- Grouped Bar Chart: Per-split distribution ----
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.25
        split_colors = {"train": "#4ECDC4", "val": "#FF6B6B", "test": "#45B7D1"}

        for idx, split in enumerate(SPLITS):
            split_data = self.stats.get("splits", {}).get(split, {}).get("classes", {})
            split_counts = [split_data.get(c, {}).get("annotation_count", 0) for c in class_names]
            ax.bar(x + idx * width, split_counts, width, label=split.capitalize(),
                   color=split_colors.get(split, "#999"), edgecolor="white")

        ax.set_title("Per-Split Class Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Defect Class", fontsize=12)
        ax.set_ylabel("Annotations", fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        ax.legend()
        ax.set_facecolor("#f8f9fa")
        plt.tight_layout()
        grouped_path = GRAPHS_DIR / "class_distribution_per_split.png"
        fig.savefig(str(grouped_path), dpi=150)
        plt.close(fig)
        logger.info(f"  Per-split chart saved: {grouped_path}")

    # ------------------------------------------------------------------
    # Markdown Report
    # ------------------------------------------------------------------
    def generate_report(self) -> Path:
        """Export summary report as Markdown."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "class_distribution.md"

        overall = self.stats.get("overall", {})
        classes = overall.get("classes", {})

        lines = [
            "# Class Distribution Report",
            "",
            f"> Generated: {self.stats['timestamp']}",
            f"> Total Annotations: **{overall.get('total_annotations', 0)}**",
            "",
            "---",
            "",
            "## Overall Class Distribution",
            "",
            "| Class | Annotations | Images | Percentage |",
            "|-------|-------------|--------|------------|",
        ]

        for cls_name, info in classes.items():
            lines.append(
                f"| {cls_name} | {info['annotation_count']} | "
                f"{info['image_count']} | {info['percentage']}% |"
            )

        # Per-split details
        for split in SPLITS:
            split_info = self.stats.get("splits", {}).get(split, {})
            split_classes = split_info.get("classes", {})
            missing = split_info.get("missing_labels", 0)
            empty = split_info.get("empty_annotations", 0)

            lines += [
                "",
                f"## {split.capitalize()} Split",
                "",
                f"- Missing labels: **{missing}**",
                f"- Empty annotation files: **{empty}**",
                "",
                "| Class | Annotations | Images |",
                "|-------|-------------|--------|",
            ]
            for cls_name, info in split_classes.items():
                lines.append(
                    f"| {cls_name} | {info['annotation_count']} | {info['image_count']} |"
                )

        # Charts
        lines += [
            "",
            "## Visualizations",
            "",
            "- Bar chart: `reports/graphs/class_distribution_bar.png`",
            "- Pie chart: `reports/graphs/class_distribution_pie.png`",
            "- Per-split chart: `reports/graphs/class_distribution_per_split.png`",
            "",
            "---",
            "*Report generated by `dataset_stats.py`*",
        ]

        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Report saved to {report_path}")
        return report_path

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full analysis pipeline."""
        logger.info("=" * 60)
        logger.info("CLASS DISTRIBUTION ANALYSIS")
        logger.info("=" * 60)

        self.scan_labels()
        self.compute_statistics()
        self.generate_charts()
        self.generate_report()

        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)


def main():
    analyzer = ClassDistributionAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
