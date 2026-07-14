# YOLOv8 Training Hyperparameter Recommendations

Based on our Week 2, Day 4 hyperparameter tuning runs comparing **SGD (lr=0.01)** and **Adam (lr=0.001)**, we recommend the following production training configurations for the Real-Time Industrial Defect Detection System.

---

## 📋 Recommended Production Configuration

| Parameter | Recommended Value | Rationale |
|---|---|---|
| **Optimizer** | **Adam** | Adam shows a `+0.0312` mAP@0.5 and `+0.0286` mAP@0.5:0.95 improvement over SGD in short training checks, demonstrating faster convergence. |
| **Learning Rate** | **0.001** | The adaptive learning rates in Adam are optimal for resolving fine, low-contrast defects (crazing and scratches) without manual LR decay scheduling. |
| **Batch Size** | **16** | A batch size of 16 balances GPU memory utilization and gradient estimation accuracy, ensuring stable training steps on edge hardware. |
| **Image Resolution** | **640x640** | The model must be trained at full 640x640 resolution to capture microscopic anomalies (e.g. pitted surfaces or scales) on the steel surface. |
| **Patience** | **50 epochs** | Prevents premature early stopping during flat validation loss plateaus, letting Adam adaptively converge. |

---

## 🔍 Detailed Parameter Analysis

### 1. Optimizer Selection: Adam vs. SGD
* **SGD (lr=0.01)**: Achieved a higher Precision of `0.5904`, but suffered on Recall (`0.4387`). This means SGD is conservative in predicting bounding boxes, missing some critical anomalies.
* **Adam (lr=0.001)**: Yielded a significantly higher Recall of `0.4701` and overall mAP@0.5 of `0.4390` (+3.12% improvement). The adaptive momentum of Adam helps in escaping saddle points in complex surface defect loss landscapes.

### 2. Learning Rate (LR) Schedule
* **Adam LR**: Recommend starting with an initial learning rate of `0.001` (default for Adam) and a linear final LR fraction of `0.01` to smoothly decay learning rates over 100 epochs.
* **Warmup**: Retain a warmup period of 3 epochs to stabilize the initial gradients while the classification layer weights initialize.

### 3. Batch Size Considerations
* While larger batches (32 or 64) speed up GPU throughput, they reduce stochastic noise in gradients. A batch size of 16 provides the optimal balance, preventing memory saturation on edge GPUs (e.g., NVIDIA Jetson Xavier / Orin).
