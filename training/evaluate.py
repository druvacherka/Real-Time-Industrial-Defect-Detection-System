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
from utils.metrics import generate_markdown_report

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
        default=str(ProjectConfig.CONFIG_DIR / "data.yaml"),
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
    
    # 2. Initialize Model
    logger.info("Initializing model for evaluation...")
    try:
        best_pt_fallback = ProjectConfig.ROOT_DIR / "results" / "train_yolov8n" / "weights" / "best.pt"
        if weights_path.exists():
            model = YOLO(str(weights_path))
            logger.info("Custom model loaded successfully.")
        elif best_pt_fallback.exists():
            weights_path = best_pt_fallback
            model = YOLO(str(weights_path))
            logger.info(f"Custom model loaded from default train run weights: {weights_path}")
        else:
            logger.warning(f"Trained weights not found at {weights_path} or fallback {best_pt_fallback}. Loading baseline model yolov8n.pt for metrics check.")
            weights_path = Path("yolov8n.pt")
            model = YOLO(str(weights_path))
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
        
    # 3. Model Validation / Metrics
    logger.info("Starting model evaluation run...")
    try:
        metrics = model.val(
            data=str(data_yaml_path),
            split=args.split,
            device=device,
            project=str(ProjectConfig.ROOT_DIR / "results"),
            name=f"evaluate_{args.split}",
            exist_ok=True
        )
        map50 = metrics.box.map50
        map95 = metrics.box.map
        logger.info(f"Evaluation completed. Metrics: mAP50={map50:.4f}, mAP50-95={map95:.4f}")
        
        # Compile metrics and generate markdown report
        try:
            names = model.names
            metrics_dict = {
                "overall": {
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(map50),
                    "map95": float(map95)
                },
                "classes": {}
            }
            
            logger.info("--- Class-wise Defect Detection Performance ---")
            
            # Extract class-wise metrics
            for i in range(len(metrics.box.all_ap)):
                class_name = names.get(i, f"class_{i}")
                # class_result returns (precision, recall, map50, map95)
                res = metrics.box.class_result(i)
                precision_val = float(res[0])
                recall_val = float(res[1])
                map50_val = float(res[2])
                map95_val = float(res[3])
                
                metrics_dict["classes"][i] = {
                    "name": class_name,
                    "precision": precision_val,
                    "recall": recall_val,
                    "map50": map50_val,
                    "map95": map95_val
                }
                
                logger.info(
                    f"Class {i} ({class_name}): "
                    f"Precision={precision_val:.4f}, "
                    f"Recall={recall_val:.4f}, "
                    f"mAP50={map50_val:.4f}, "
                    f"mAP50-95={map95_val:.4f}"
                )
            logger.info("------------------------------------------------")
            
            report_path = ProjectConfig.ROOT_DIR / "results" / f"evaluation_report_{args.split}.md"
            generate_markdown_report(metrics_dict, report_path)
            logger.info(f"Automatically generated and saved markdown report to {report_path}")
        except Exception as report_err:
            logger.warning(f"Could not generate evaluation markdown report: {report_err}")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        sys.exit(1)
        
    params = {
        "weights": str(weights_path),
        "data": str(data_yaml_path),
        "split": args.split,
        "device": str(device),
        "map50": map50,
        "map95": map95
    }
    return params


if __name__ == "__main__":
    main()
