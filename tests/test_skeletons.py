"""
Unit tests to verify CLI arguments and configurations of train, predict, and evaluate skeletons.
"""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train import main as train_main
from training.predict import main as predict_main
from training.evaluate import main as evaluate_main


class TestSkeletons(unittest.TestCase):
    
    @patch('training.train.YOLO.train')
    def test_train_skeleton_args(self, mock_train):
        """Tests that train.py parses configurations and overrides correctly."""
        test_args = [
            "train.py",
            "--epochs", "50",
            "--batch_size", "8",
            "--device", "cpu",
            "--model", "yolov8m"
        ]
        with patch.object(sys, "argv", test_args):
            params = train_main()
            self.assertEqual(params["epochs"], 50)
            self.assertEqual(params["batch_size"], 8)
            self.assertEqual(params["device"], "cpu")
            self.assertEqual(params["model_arch"], "yolov8m")
            mock_train.assert_called_once()
            
    def test_predict_skeleton_args(self):
        """Tests that predict.py parses arguments correctly."""
        test_args = [
            "predict.py",
            "--source", "datasets/raw/NEU-DET/train/images/crazing/crazing_1.jpg",
            "--conf", "0.5",
            "--device", "cpu"
        ]
        with patch.object(sys, "argv", test_args):
            params = predict_main()
            self.assertEqual(params["conf"], 0.5)
            self.assertEqual(params["device"], "cpu")
            self.assertTrue(params["source"].endswith("crazing_1.jpg"))
            
    def test_evaluate_skeleton_args(self):
        """Tests that evaluate.py parses arguments correctly."""
        test_args = [
            "evaluate.py",
            "--split", "test",
            "--device", "cpu"
        ]
        with patch.object(sys, "argv", test_args):
            params = evaluate_main()
            self.assertEqual(params["split"], "test")
            self.assertEqual(params["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
