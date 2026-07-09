"""
Orchestration script to run model evaluation sequentially over Val and Test splits
and generate a side-by-side comparative report.
====================================================================================
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.device import get_device
from utils.logger import get_logger
from utils.metrics import generate_splits_comparison_report

logger = get_logger("splits_comparison")


def extract_metrics(metrics, model) -> dict:
    """Helper to compile validation results into metrics dictionary."""
    names = model.names
    metrics_dict = {
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map95": float(metrics.box.map)
        },
        "classes": {}
    }
    # Extract class-wise metrics
    for i, class_name in names.items():
        res = metrics.box.class_result(i)
        metrics_dict["classes"][i] = {
            "name": class_name,
            "precision": float(res[0]),
            "recall": float(res[1]),
            "map50": float(res[2]),
            "map95": float(res[3])
        }
    return metrics_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model performance on both val and test splits.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"),
        help="Path to trained YOLOv8 weights."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ProjectConfig.CONFIG_DIR / "data.yaml"),
        help="Path to dataset configuration YAML."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu', 'cuda', etc.)."
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    data_yaml_path = Path(args.data)
    results_dir = ProjectConfig.ROOT_DIR / "results"
    
    # 1. Resolve weights path
    if not weights_path.exists():
        logger.warning(f"Trained weights not found at {weights_path}. Falling back to baseline yolov8n.pt.")
        weights_path = Path("yolov8n.pt")
        if not weights_path.exists():
            # Trigger download
            YOLO("yolov8n.pt")

    logger.info(f"Loading YOLOv8 model weights: {weights_path}")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

    device = get_device() if args.device == "auto" else args.device

    # 2. Run Validation Split Evaluation
    logger.info("--- [1/2] Running Validation Split Evaluation ---")
    try:
        val_results = model.val(
            data=str(data_yaml_path),
            split="val",
            device=device,
            project=str(results_dir),
            name="evaluate_compare_val",
            exist_ok=True
        )
        val_metrics = extract_metrics(val_results, model)
    except Exception as e:
        logger.error(f"Error during validation split evaluation: {e}")
        sys.exit(1)

    # 3. Run Test Split Evaluation
    logger.info("--- [2/2] Running Test Split Evaluation ---")
    try:
        test_results = model.val(
            data=str(data_yaml_path),
            split="test",
            device=device,
            project=str(results_dir),
            name="evaluate_compare_test",
            exist_ok=True
        )
        test_metrics = extract_metrics(test_results, model)
    except Exception as e:
        logger.error(f"Error during test split evaluation: {e}")
        sys.exit(1)

    # 4. Generate Comparative Report
    report_path = results_dir / "splits_comparison_report.md"
    logger.info(f"Generating splits comparative report: {report_path}")
    try:
        generate_splits_comparison_report(val_metrics, test_metrics, report_path)
        logger.info("Splits comparative metrics report generated successfully!")
    except Exception as e:
        logger.error(f"Error generating comparison report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
