"""
Unit tests to verify prediction and evaluation script execution flows.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.predict import main as predict_main
from training.evaluate import main as evaluate_main


class TestInferencePipeline(unittest.TestCase):
    
    @patch('training.predict.YOLO')
    def test_predict_execution_flow(self, mock_yolo_class):
        """Verifies predict.py main execution calls YOLO predict with correct parameters."""
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance
        
        test_args = [
            "predict.py",
            "--weights", "yolov8n.pt",
            "--source", "dataset/yolo/images/test/crazing_101.jpg",
            "--conf", "0.35",
            "--device", "cpu"
        ]
        
        with patch.object(sys, "argv", test_args):
            params = predict_main()
            
            # Assertions
            self.assertEqual(params["weights"], "yolov8n.pt")
            self.assertEqual(params["conf"], 0.35)
            self.assertEqual(params["device"], "cpu")
            self.assertTrue(params["source"].endswith("crazing_101.jpg"))
            
            # Verify mock YOLO predict was invoked
            mock_yolo_instance.predict.assert_called_once()
            call_kwargs = mock_yolo_instance.predict.call_args[1]
            self.assertEqual(call_kwargs["conf"], 0.35)
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertEqual(call_kwargs["save"], True)
            
    @patch('training.evaluate.YOLO')
    def test_evaluate_execution_flow(self, mock_yolo_class):
        """Verifies evaluate.py main execution calls YOLO val with correct parameters."""
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance
        
        # Mock returned validation metrics
        mock_metrics = MagicMock()
        mock_metrics.box.map50 = 0.85
        mock_metrics.box.map = 0.55
        mock_yolo_instance.val.return_value = mock_metrics
        
        test_args = [
            "evaluate.py",
            "--weights", "yolov8n.pt",
            "--split", "val",
            "--device", "cpu"
        ]
        
        with patch.object(sys, "argv", test_args):
            params = evaluate_main()
            
            # Assertions
            self.assertEqual(params["weights"], "yolov8n.pt")
            self.assertEqual(params["split"], "val")
            self.assertEqual(params["device"], "cpu")
            self.assertEqual(params["map50"], 0.85)
            self.assertEqual(params["map95"], 0.55)
            
            # Verify mock YOLO val was invoked
            mock_yolo_instance.val.assert_called_once()
            call_kwargs = mock_yolo_instance.val.call_args[1]
            self.assertEqual(call_kwargs["split"], "val")
            self.assertEqual(call_kwargs["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
