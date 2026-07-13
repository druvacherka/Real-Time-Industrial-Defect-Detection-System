# YOLOv8 Baseline Model Evaluation Performance Report

This document reports the performance metrics of the baseline YOLOv8 model (`train_yolov8n` run) trained on the NEU Metal Surface Defects dataset.

---

## 📊 Overall Model Metrics

The table below displays the overall Precision, Recall, and mAP metrics compiled on both the **Validation** and **Test** splits, along with the performance gap (Delta).

| Metric | Validation Split | Test Split | Delta (Test - Val) |
|---|---|---|---|
| **PRECISION** | 0.5273 | 0.4958 | -0.0315 |
| **RECALL** | 0.4291 | 0.4332 | +0.0042 |
| **mAP@0.5** | 0.4075 | 0.3963 | -0.0113 |
| **mAP@0.5:0.95** | 0.1867 | 0.1820 | -0.0046 |

### Analysis
* **Generalization**: The baseline model exhibits excellent generalization, with minimal performance drop on the unseen test split. The mAP@0.5 difference is only `-0.0113` (less than 1.2% absolute delta).
* **Precision vs. Recall**: The model has higher overall precision (`0.5273`) than recall (`0.4291`), indicating that detections are generally accurate, but some defects are missed.

---

## 📋 Class-wise Defect Detection Performance

Below is a detailed breakdown of metrics for the six defect categories.

### 1. Validation Split Performance

| Class ID | Class Name | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **0** | crazing | 1.0000 | 0.0000 | 0.0377 | 0.0081 |
| **1** | inclusion | 0.6533 | 0.3510 | 0.4407 | 0.1526 |
| **2** | patches | 0.6430 | 0.6995 | 0.7397 | 0.3884 |
| **3** | pitted_surface | 0.3099 | 0.6739 | 0.6304 | 0.3588 |
| **4** | rolled-in_scale | 0.0905 | 0.2119 | 0.0491 | 0.0107 |
| **5** | scratches | 0.4670 | 0.6381 | 0.5477 | 0.2014 |

### 2. Test Split Performance

| Class ID | Class Name | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **0** | crazing | 1.0000 | 0.0000 | 0.0488 | 0.0104 |
| **1** | inclusion | 0.5530 | 0.4260 | 0.4520 | 0.1740 |
| **2** | patches | 0.4790 | 0.5160 | 0.5630 | 0.3030 |
| **3** | pitted_surface | 0.3360 | 0.6810 | 0.6320 | 0.3620 |
| **4** | rolled-in_scale | 0.1170 | 0.3470 | 0.1270 | 0.0341 |
| **5** | scratches | 0.4910 | 0.6290 | 0.5550 | 0.2080 |

### Defect Class Performance Summary
1. **Patches & Pitted Surfaces**: The model achieves high mAP@0.5 on `patches` (`0.7397`) and `pitted_surface` (`0.6304`). These defects have distinct spatial structures and are easiest for the model to localize.
2. **Scratches & Inclusions**: Scratches (`0.5477` mAP@0.5) and inclusions (`0.4407` mAP@0.5) show moderate, acceptable baseline performance.
3. **Crazing & Rolled-in Scale**: Crazing (`0.0377` mAP@0.5) and rolled-in scale (`0.0491` mAP@0.5) are the hardest categories. Crazing has extremely fine, low-contrast cracks, and rolled-in scale has amorphous boundaries, requiring hyperparameter tuning and longer training epochs to improve convergence.
