"""
End-to-End Dry-Run Verification Script for real-time defect detection pipeline.
Orchestrates: training dry-run -> validation/evaluation -> prediction.
"""

import sys
from pathlib import Path
import time

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import main as train_main
from training.evaluate import main as evaluate_main
from training.predict import main as predict_main
from utils.logger import get_logger

logger = get_logger("dry_run_pipeline")


def run_pipeline():
    logger.info("=" * 60)
    logger.info("STARTING END-TO-END DRY-RUN PIPELINE VERIFICATION")
    logger.info("=" * 60)
    
    # Step 1: Run 1-epoch dry-run training
    logger.info("[STEP 1/3] Triggering 1-epoch model training dry-run...")
    sys.argv = ["train.py", "--dry-run", "--device", "cpu"]
    train_params = train_main()
    
    trained_weights = Path(train_params["results_dir"]) / train_params["run_name"] / "weights" / "best.pt"
    logger.info(f"Step 1 Complete. Trained weights generated at: {trained_weights}")
    logger.info("-" * 60)
    
    # Step 2: Run validation evaluation on the generated weights
    logger.info("[STEP 2/3] Triggering validation split evaluation using dry-run weights...")
    sys.argv = ["evaluate.py", "--weights", str(trained_weights), "--split", "val", "--device", "cpu"]
    eval_params = evaluate_main()
    logger.info(f"Step 2 Complete. Evaluation metrics: mAP50={eval_params['map50']:.4f}, mAP50-95={eval_params['map95']:.4f}")
    logger.info("-" * 60)
    
    # Step 3: Run inference prediction on a sample image
    sample_image = PROJECT_ROOT / "dataset" / "yolo" / "images" / "test" / "crazing_101.jpg"
    logger.info(f"[STEP 3/3] Running prediction inference on sample image: {sample_image}")
    sys.argv = ["predict.py", "--weights", str(trained_weights), "--source", str(sample_image), "--device", "cpu"]
    predict_params = predict_main()
    logger.info(f"Step 3 Complete. Visualized predictions saved under: {predict_params['save_dir']}")
    
    logger.info("=" * 60)
    logger.info("END-TO-END DRY-RUN PIPELINE SUCCESSFUL!")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
