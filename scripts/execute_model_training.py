"""
Orchestration script to execute, monitor, and verify the model training pipeline.
====================================================================================
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("training_executor")


def main():
    parser = argparse.ArgumentParser(description="Execute and monitor YOLOv8 model training.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ProjectConfig.CONFIG_DIR / "experiment.yaml"),
        help="Path to the experiment configuration YAML."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        help="Override image size."
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use ('cpu', 'cuda', etc.)."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override YOLOv8 model architecture (e.g. yolov8n, yolov8s)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Override early stopping patience epochs."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (1 epoch, batch size 2, minimal workers)."
    )
    args = parser.parse_args()

    # Construct python command
    python_exe = sys.executable
    train_script = ProjectConfig.ROOT_DIR / "training" / "train.py"
    
    cmd = [python_exe, str(train_script), "--config", args.config]
    
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.img_size is not None:
        cmd.extend(["--img_size", str(args.img_size)])
    if args.device is not None:
        cmd.extend(["--device", args.device])
    if args.model is not None:
        cmd.extend(["--model", args.model])
    if args.patience is not None:
        cmd.extend(["--patience", str(args.patience)])
    if args.resume:
        cmd.append("--resume")
    if args.dry_run:
        cmd.append("--dry-run")

    logger.info(f"Launching training script with command: {' '.join(cmd)}")
    try:
        # Run training subprocess and pipe output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        
        # Print output to logs in real-time
        for line in process.stdout:
            try:
                print(line.rstrip())
            except UnicodeEncodeError:
                # Fallback to replacing unencodable characters for the terminal
                out_enc = sys.stdout.encoding or "utf-8"
                cleaned = line.encode(out_enc, errors="replace").decode(out_enc)
                print(cleaned.rstrip())
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Training script failed with exit code: {process.returncode}")
            sys.exit(process.returncode)
            
        logger.info("Training script completed successfully. Verifying output artifacts...")
        
        # Verify outputs
        model_arch = args.model if args.model is not None else "yolov8n"
        results_dir = ProjectConfig.ROOT_DIR / "results"
        run_name = f"train_{model_arch}"
        
        weights_path = results_dir / run_name / "weights" / "best.pt"
        loss_curves = results_dir / run_name / "loss_curves.png"
        map_curves = results_dir / run_name / "map_curves.png"
        
        if not weights_path.exists():
            logger.error(f"Expected weights not found at: {weights_path}")
            sys.exit(1)
            
        logger.info(f"Verified trained weights present at: {weights_path}")
        
        if loss_curves.exists() and map_curves.exists():
            logger.info("Verified training performance visualization curves generated successfully.")
        else:
            logger.warning("Visualization curves were not found in the run directory.")
            
        logger.info("=== Model Training Run Validation Successful ===")

    except Exception as e:
        logger.error(f"Error during training execution orchestration: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
