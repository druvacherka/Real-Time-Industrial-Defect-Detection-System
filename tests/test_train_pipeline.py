"""
Unit test verifying model training execution pipeline in dry-run mode.
"""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path
import shutil

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import main as train_main


class TestTrainPipeline(unittest.TestCase):
    
    def test_train_dry_run_execution(self):
        """Runs the YOLO training pipeline in dry-run mode and checks checkpoint generation."""
        # Setup dry run args
        test_args = [
            "train.py",
            "--dry-run",
            "--device", "cpu"
        ]
        
        with patch.object(sys, "argv", test_args):
            params = train_main()
            
            # Verify parameters returned by main
            self.assertEqual(params["epochs"], 1)
            self.assertEqual(params["batch_size"], 2)
            self.assertEqual(params["img_size"], 64)
            self.assertEqual(params["device"], "cpu")
            
            # Verify run directory and weights output
            results_dir = Path(params["results_dir"])
            run_name = params["run_name"]
            run_dir = results_dir / run_name
            weights_dir = run_dir / "weights"
            
            self.assertTrue(run_dir.exists(), f"Run directory {run_dir} not created.")
            self.assertTrue(weights_dir.exists(), f"Weights directory {weights_dir} not created.")
            
            # Clean up the dry-run results to prevent clogging the workspace
            try:
                shutil.rmtree(run_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up dry-run results directory {run_dir}: {e}")


if __name__ == "__main__":
    unittest.main()
