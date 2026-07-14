"""
Orchestration script to run hyperparameter tuning experiments, compare metrics, and output reports.
====================================================================================================
"""

import sys
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("hyperparameter_tuning")


def parse_run_results(run_dir: Path) -> dict:
    """Parses results.csv for the given run directory and extracts final metrics."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        logger.error(f"Results CSV not found at: {csv_path}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        
        # Get col name mappings
        p_col = [c for c in df.columns if "precision" in c][0]
        r_col = [c for c in df.columns if "recall" in c][0]
        map50_col = [c for c in df.columns if "mAP50(B)" in c or "mAP50" in c][0]
        map95_col = [c for c in df.columns if "mAP50-95(B)" in c or "mAP50-95" in c][0]
        
        return {
            "precision": float(last_row[p_col]),
            "recall": float(last_row[r_col]),
            "map50": float(last_row[map50_col]),
            "map95": float(last_row[map95_col]),
        }
    except Exception as e:
        logger.error(f"Error parsing results at {csv_path}: {e}")
        return {}


def main():
    logger.info("Initializing hyperparameter tuning configurations...")
    
    data_yaml_path = ProjectConfig.CONFIG_DIR / "data.yaml"
    results_dir = ProjectConfig.ROOT_DIR / "results"
    
    # Configurations to compare
    configs = [
        {
            "name": "SGD_lr_0.01",
            "run_name": "tune_sgd",
            "optimizer": "SGD",
            "lr0": 0.01,
            "epochs": 3,
            "imgsz": 128,
            "batch": 16,
        },
        {
            "name": "Adam_lr_0.001",
            "run_name": "tune_adam",
            "optimizer": "Adam",
            "lr0": 0.001,
            "epochs": 3,
            "imgsz": 128,
            "batch": 16,
        }
    ]
    
    metrics_summary = {}
    
    for cfg in configs:
        logger.info(f"=== Starting Training Run for Configuration: {cfg['name']} ===")
        try:
            # Re-init YOLO base weights
            model = YOLO("yolov8n.pt")
            
            model.train(
                data=str(data_yaml_path),
                epochs=cfg["epochs"],
                batch=cfg["batch"],
                imgsz=cfg["imgsz"],
                device="cpu",
                optimizer=cfg["optimizer"],
                lr0=cfg["lr0"],
                project=str(results_dir),
                name=cfg["run_name"],
                exist_ok=True,
                workers=2,
            )
            
            run_dir = results_dir / cfg["run_name"]
            metrics = parse_run_results(run_dir)
            if metrics:
                metrics_summary[cfg["name"]] = metrics
                logger.info(f"Completed run {cfg['name']}. Final metrics: {metrics}")
            else:
                logger.error(f"Could not extract metrics for {cfg['name']}")
                
        except Exception as e:
            logger.error(f"Error executing config {cfg['name']}: {e}")
            
    # Compile comparison report
    if len(metrics_summary) == 2:
        logger.info("Comparing metric changes across parameter sets...")
        cfg1_name = configs[0]["name"]
        cfg2_name = configs[1]["name"]
        
        m1 = metrics_summary[cfg1_name]
        m2 = metrics_summary[cfg2_name]
        
        # Calculate Deltas (Config 2 - Config 1)
        deltas = {k: m2[k] - m1[k] for k in m1.keys()}
        
        # Print comparison to console
        logger.info("=== Hyperparameter Comparison Summary ===")
        logger.info(f"Metric | {cfg1_name} | {cfg2_name} | Delta")
        for k in m1.keys():
            logger.info(f"{k.upper()}: {m1[k]:.4f} vs {m2[k]:.4f} (Delta: {deltas[k]:+.4f})")
        logger.info("==========================================")
        
        # Generate comparative Markdown report
        report_path = results_dir / "hyperparameter_tuning_report.md"
        report_content = f"""# Hyperparameter Tuning Comparative Report

This report compares model performance across different hyperparameter configurations to recommend the optimal settings for production.

## Configurations Tested

1. **Configuration 1 (SGD)**:
   * Optimizer: SGD
   * Initial Learning Rate (lr0): 0.01
   * Batch Size: 16
   * Resolution: 128x128
   * Epochs: 3

2. **Configuration 2 (Adam)**:
   * Optimizer: Adam
   * Initial Learning Rate (lr0): 0.001
   * Batch Size: 16
   * Resolution: 128x128
   * Epochs: 3

## Comparative Metrics Summary

| Metric | Configuration 1 (SGD) | Configuration 2 (Adam) | Delta (Adam - SGD) |
|---|---|---|---|
| **Precision** | {m1['precision']:.4f} | {m2['precision']:.4f} | {deltas['precision']:+.4f} |
| **Recall** | {m1['recall']:.4f} | {m2['recall']:.4f} | {deltas['recall']:+.4f} |
| **mAP@0.5** | {m1['map50']:.4f} | {m2['map50']:.4f} | {deltas['map50']:+.4f} |
| **mAP@0.5:0.95** | {m1['map95']:.4f} | {m2['map95']:.4f} | {deltas['map95']:+.4f} |

## Findings and Analysis
* The comparative evaluation checks the change in validation accuracy across Adam vs SGD.
* Adam optimizer with learning rate 0.001 typically demonstrates smoother convergence on fine defect details (e.g. crazing/rolled-in scale) due to adaptive learning rates, while SGD provides robust baseline gradient steps.
"""
        report_path.write_text(report_content, encoding="utf-8")
        logger.info(f"Successfully generated tuning report at: {report_path}")
    else:
        logger.warning("Could not generate comparative report due to missing metrics.")


if __name__ == "__main__":
    main()
