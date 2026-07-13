# Hyperparameter Tuning Comparative Report

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
| **Precision** | 0.5904 | 0.5551 | -0.0354 |
| **Recall** | 0.4387 | 0.4701 | +0.0313 |
| **mAP@0.5** | 0.4078 | 0.4390 | +0.0312 |
| **mAP@0.5:0.95** | 0.1821 | 0.2107 | +0.0286 |

## Findings and Analysis
* The comparative evaluation checks the change in validation accuracy across Adam vs SGD.
* Adam optimizer with learning rate 0.001 typically demonstrates smoother convergence on fine defect details (e.g. crazing/rolled-in scale) due to adaptive learning rates, while SGD provides robust baseline gradient steps.
