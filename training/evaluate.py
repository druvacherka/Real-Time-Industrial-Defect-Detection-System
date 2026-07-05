"""
YOLOv8 Evaluation and Validation Script Skeleton.
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

logger = get_logger("evaluation")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model performance on NEU split.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ProjectConfig.WEIGHTS_DIR / "best.pt"),
        help="Path to trained YOLOv8 weights."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(ProjectConfig.DATASET_DIR / "data.yaml"),
        help="Path to dataset configuration YAML."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on ('val' or 'test')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use ('cpu', 'cuda', etc.)."
    )
    
    args = parser.parse_args()
    
    # 1. Verify paths
    weights_path = Path(args.weights)
    data_yaml_path = Path(args.data)
    
    logger.info(f"Evaluating model weights: {weights_path}")
    logger.info(f"Dataset configuration: {data_yaml_path}")
    logger.info(f"Split: {args.split}")
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
        
    logger.info(f"Using device: {device}")
    
    # 2. Initialize Model Skeleton
    logger.info("Initializing model for evaluation...")
    try:
        if weights_path.exists():
            model = YOLO(str(weights_path))
            logger.info("Custom model loaded successfully.")
        else:
            logger.warn(f"Trained weights {weights_path} not found. Loading baseline model yolov8n.pt for metrics check.")
            model = YOLO("yolov8n.pt")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Model Validation / Metrics Skeleton
    logger.info("Starting model evaluation run...")
    # metrics = model.val(data=str(data_yaml_path), split=args.split, device=device)
    logger.info("Evaluation pipeline skeleton ready. Waiting for custom trained model.")
    
    params = {
        "weights": str(weights_path),
        "data": str(data_yaml_path),
        "split": args.split,
        "device": str(device)
    }
    return params


if __name__ == "__main__":
    main()
