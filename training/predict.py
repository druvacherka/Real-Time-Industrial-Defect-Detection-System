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
    
    # 2. Initialize Model
    logger.info("Initializing prediction model...")
    try:
        # Check if weights file exists, otherwise fallback to finding it in results/train_yolov8n/weights/best.pt
        best_pt_fallback = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
        if weights_path.exists():
            model = YOLO(str(weights_path))
            logger.info("Custom model loaded successfully.")
        elif best_pt_fallback.exists():
            weights_path = best_pt_fallback
            model = YOLO(str(weights_path))
            logger.info(f"Custom model loaded from default train run weights: {weights_path}")
        else:
            logger.warning(f"Trained weights not found at {weights_path} or fallback {best_pt_fallback}. Initializing with default yolov8n.pt.")
            weights_path = Path("yolov8n.pt")
            model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        sys.exit(1)
        
    # 3. Model Prediction
    logger.info(f"Starting inference with conf threshold {args.conf}...")
    save_dir = ProjectConfig.ROOT_DIR / "results" / "predictions"
    try:
        results = model.predict(
            source=str(source_path),
            conf=args.conf,
            device=device,
            project=str(ProjectConfig.ROOT_DIR / "results"),
            name="predictions",
            save=True,
            exist_ok=True
        )
        logger.info(f"Prediction pipeline completed. Visualized outputs saved to {save_dir}")
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        sys.exit(1)
    
    params = {
        "weights": str(weights_path),
        "source": str(source_path),
        "conf": args.conf,
        "device": str(device),
        "save_dir": str(save_dir)
    }
    return params


if __name__ == "__main__":
    main()
