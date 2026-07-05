"""
YOLOv8 Inference / Prediction Script Skeleton.
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

logger = get_logger("prediction")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on defect images.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ProjectConfig.WEIGHTS_DIR / "best.pt"),
        help="Path to trained YOLOv8 weights."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source image, video, or folder."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections."
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
    source_path = Path(args.source)
    
    logger.info(f"Loading weights from: {weights_path}")
    logger.info(f"Source input: {source_path}")
    
    if not source_path.exists():
        logger.error(f"Inference source not found: {source_path}")
        sys.exit(1)
        
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
        
    logger.info(f"Using device: {device}")
    
    # 2. Initialize Model Skeleton
    # Note: On Day 3, we mock model initialization if the best.pt weights do not exist yet.
    logger.info("Initializing prediction model...")
    try:
        if weights_path.exists():
            model = YOLO(str(weights_path))
            logger.info("Custom model loaded successfully.")
        else:
            logger.warn(f"Trained weights {weights_path} not found. Initializing with default yolov8n.pt.")
            model = YOLO("yolov8n.pt")
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        sys.exit(1)
        
    # 3. Model Prediction Skeleton
    logger.info(f"Starting inference with conf threshold {args.conf}...")
    # results = model.predict(source=str(source_path), conf=args.conf, device=device)
    logger.info("Prediction pipeline skeleton ready. Waiting for trained model weights.")
    
    params = {
        "weights": str(weights_path),
        "source": str(source_path),
        "conf": args.conf,
        "device": str(device)
    }
    return params


if __name__ == "__main__":
    main()
