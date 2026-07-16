"""
ONNX Model Export Script.
=========================
Converts custom YOLOv8 model weights (.pt) to ONNX format for hardware integration.
"""

import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("onnx_export")


def main():
    logger.info("Starting model conversion to ONNX...")
    
    weights_path = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
    fallback_path = ProjectConfig.ROOT_DIR / "yolov8n.pt"
    
    if weights_path.exists():
        model_load_path = weights_path
        logger.info(f"Loading custom trained weights from: {model_load_path}")
    else:
        model_load_path = fallback_path
        logger.warning(f"Custom trained weights not found at {weights_path}. Exporting base {fallback_path}...")
        if not fallback_path.exists():
            YOLO("yolov8n.pt")  # Download baseline model
            
    try:
        model = YOLO(str(model_load_path))
        
        # Exporting YOLOv8 model to ONNX format
        logger.info("Executing Ultralytics ONNX exporter...")
        exported_path_str = model.export(format="onnx", imgsz=640, dynamic=True)
        exported_path = Path(exported_path_str)
        
        exports_dir = ProjectConfig.ROOT_DIR / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = exports_dir / "model.onnx"
        shutil.copy(exported_path, dest_path)
        
        logger.info(f"Model exported successfully to custom location: {dest_path}")
    except Exception as e:
        logger.error(f"Error during ONNX export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
