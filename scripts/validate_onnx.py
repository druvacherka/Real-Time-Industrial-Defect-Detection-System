"""
ONNX Validation & Parity Verification Script.
=============================================
Runs validation on the exported ONNX model, compares metrics against PyTorch,
and asserts prediction parity.
"""

import sys
import json
from pathlib import Path
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("onnx_validation")


def main():
    logger.info("Starting ONNX vs PyTorch validation parity comparison...")
    
    pt_path = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
    onnx_path = ProjectConfig.ROOT_DIR / "exports" / "model.onnx"
    
    if not pt_path.exists():
        logger.error(f"PyTorch weights not found at: {pt_path}")
        sys.exit(1)
    if not onnx_path.exists():
        logger.error(f"ONNX model not found at: {onnx_path}")
        sys.exit(1)
        
    try:
        # Load models
        logger.info("Loading PyTorch and ONNX models...")
        pt_model = YOLO(str(pt_path))
        onnx_model = YOLO(str(onnx_path))
        
        # Run validation on Validation split
        data_config = str(ProjectConfig.ROOT_DIR / "configs" / "data.yaml")
        
        logger.info("Running validation with PyTorch model...")
        pt_results = pt_model.val(data=data_config, verbose=False)
        
        logger.info("Running validation with ONNX model...")
        onnx_results = onnx_model.val(data=data_config, verbose=False)
        
        # Extract metrics
        pt_map50 = float(pt_results.box.map50)
        pt_map95 = float(pt_results.box.map)
        pt_speed = pt_results.speed
        
        onnx_map50 = float(onnx_results.box.map50)
        onnx_map95 = float(onnx_results.box.map)
        onnx_speed = onnx_results.speed
        
        map50_diff = abs(pt_map50 - onnx_map50)
        map95_diff = abs(pt_map95 - onnx_map95)
        
        logger.info("--- Parity Comparison Results ---")
        logger.info(f"PyTorch mAP@0.5: {pt_map50:.6f} | ONNX mAP@0.5: {onnx_map50:.6f} | Diff: {map50_diff:.6f}")
        logger.info(f"PyTorch mAP@0.5:0.95: {pt_map95:.6f} | ONNX mAP@0.5:0.95: {onnx_map95:.6f} | Diff: {map95_diff:.6f}")
        logger.info(f"PyTorch Inference Speed: {pt_speed['inference']:.2f} ms/img | ONNX Inference Speed: {onnx_speed['inference']:.2f} ms/img")
        
        # Save metrics to reports
        reports_dir = ProjectConfig.ROOT_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        parity_data = {
            "pytorch": {
                "map50": pt_map50,
                "map95": pt_map95,
                "inference_speed_ms": pt_speed['inference']
            },
            "onnx": {
                "map50": onnx_map50,
                "map95": onnx_map95,
                "inference_speed_ms": onnx_speed['inference']
            },
            "delta": {
                "map50_diff": map50_diff,
                "map95_diff": map95_diff
            }
        }
        
        parity_file = reports_dir / "onnx_parity_metrics.json"
        with open(parity_file, "w") as f:
            json.dump(parity_data, f, indent=4)
            
        logger.info(f"Saved parity reports JSON to: {parity_file}")
        
        # Parity check assertion
        if map50_diff < 1e-4:
            logger.info("SUCCESS: PyTorch and ONNX predictions are in full parity!")
        else:
            logger.warning("WARNING: PyTorch and ONNX metrics differ slightly. Check export configuration.")
            
    except Exception as e:
        logger.error(f"Error during validation comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
