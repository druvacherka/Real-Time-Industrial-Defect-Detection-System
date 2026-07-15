# Preprocessing Pipeline Benchmark Report

Generated dynamically by `benchmark_preprocessing.py`.

## ⚙️ Test Specifications
- **Runs per Configuration**: 2000 iterations
- **Input Dimension**: 300x300x3 BGR array
- **Target Dimension**: 200x200x3

## 📊 Resize Interpolation Speeds
| Interpolation Mode | Total Execution Time (s) | Throughput (FPS) |
| --- | --- | --- |
| Nearest | 0.1204 | 16617.29 |
| Bilinear | 0.3356 | 5959.43 |
| Bicubic | 0.4481 | 4462.92 |
| Area | 0.9568 | 2090.28 |

## 🧪 Normalization Modes Comparison
| Normalization Method | Total Execution Time (s) | Throughput (FPS) |
| --- | --- | --- |
| Min-Max Scaling [0, 1] | 0.6389 | 3130.17 |
| Standard Z-Score (ImageNet) | 2.3584 | 848.02 |
