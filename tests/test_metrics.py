"""
Unit tests for the evaluation metrics visualization and reporting utilities.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import sys

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics import (
    parse_training_results,
    plot_learning_curves,
    generate_markdown_report,
)


class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create a mock YOLOv8 results.csv
        self.mock_csv_path = self.temp_dir / "results.csv"
        data = {
            "epoch": [1, 2, 3],
            "train/box_loss": [1.5, 1.2, 0.9],
            "val/box_loss": [1.4, 1.1, 0.85],
            "train/cls_loss": [2.5, 2.0, 1.6],
            "val/cls_loss": [2.3, 1.8, 1.5],
            "metrics/mAP50(B)": [0.35, 0.55, 0.72],
            "metrics/mAP50-95(B)": [0.18, 0.32, 0.45]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.mock_csv_path, index=False)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_parse_training_results(self):
        """Should parse training results CSV into DataFrame with stripped columns."""
        df = parse_training_results(self.mock_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn("epoch", df.columns)
        self.assertIn("train/box_loss", df.columns)
        
    def test_plot_learning_curves(self):
        """Should successfully plot loss and mAP curves as PNGs."""
        plot_dir = self.temp_dir / "plots"
        plot_learning_curves(self.mock_csv_path, plot_dir)
        
        loss_plot = plot_dir / "loss_curves.png"
        map_plot = plot_dir / "map_curves.png"
        
        self.assertTrue(loss_plot.exists(), "loss_curves.png was not created")
        self.assertTrue(map_plot.exists(), "map_curves.png was not created")
        
    def test_generate_markdown_report(self):
        """Should generate formatted markdown tables with class details."""
        metrics_dict = {
            "overall": {
                "precision": 0.8123,
                "recall": 0.7890,
                "map50": 0.8456,
                "map95": 0.5567
            },
            "classes": {
                0: {"name": "crazing", "precision": 0.80, "recall": 0.75, "map50": 0.82, "map95": 0.52},
                1: {"name": "inclusion", "precision": 0.82, "recall": 0.80, "map50": 0.85, "map95": 0.58}
            }
        }
        
        report_path = self.temp_dir / "report.md"
        generate_markdown_report(metrics_dict, report_path)
        
        self.assertTrue(report_path.exists(), "report.md was not created")
        
        # Verify content
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("# Model Evaluation Performance Report", content)
        self.assertIn("crazing", content)
        self.assertIn("inclusion", content)
        self.assertIn("0.8456", content)


if __name__ == "__main__":
    unittest.main()
