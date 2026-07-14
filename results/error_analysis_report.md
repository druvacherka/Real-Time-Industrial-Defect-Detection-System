# Confusion Matrix and Error Analysis Report

This report documents the baseline model's classification errors, detailing False Positives, False Negatives, and Misclassifications on the Validation split.

---

## 📊 Confusion Matrix

The table below displays the prediction-to-ground-truth mapping. Rows represent Ground Truth classes and columns represent Predicted classes.

| GT \ Pred | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches | background |
|---| --- | --- | --- | --- | --- | --- | --- |
| **crazing** | 0 | 0 | 0 | 1 | 0 | 0 | 128 |
| **inclusion** | 0 | 86 | 0 | 0 | 0 | 0 | 118 |
| **patches** | 0 | 0 | 160 | 0 | 0 | 0 | 24 |
| **pitted_surface** | 0 | 0 | 0 | 62 | 0 | 0 | 30 |
| **rolled-in_scale** | 0 | 0 | 0 | 0 | 42 | 0 | 76 |
| **scratches** | 0 | 0 | 0 | 0 | 0 | 81 | 24 |
| **background** | 0 | 32 | 85 | 97 | 139 | 122 | 0 |

*Note: background column represents False Negatives (model missed defects); background row represents False Positives (model detected defects on background).*

---

## 🔍 Error Analysis Summary

* **Total False Positives (FP)**: 476
* **Total False Negatives (FN)**: 400

### False Positives / Negatives Count by Class

| Class Name | False Positives (Background FP) | False Negatives (Model Missed) |
|---|---|---|
| crazing | 0 | 128 |
| inclusion | 32 | 118 |
| patches | 85 | 24 |
| pitted_surface | 97 | 30 |
| rolled-in_scale | 139 | 76 |
| scratches | 122 | 24 |

---

## 📋 Detailed Error Log Samples (First 15 Errors)

The table below lists individual False Positive and False Negative defect details for debugging.

| Image File | Ground Truth | Predicted | Confidence | Bounding Box [x1, y1, x2, y2] |
|---|---|---|---|---|
| crazing_288.jpg | background | pitted_surface | 0.3393 | [np.float64(0.1), np.float64(0.1), np.float64(200.0), np.float64(198.1)] |
| crazing_92.jpg | crazing | pitted_surface | 0.2740 | [np.float64(2.6), np.float64(0.0), np.float64(199.9), np.float64(100.3)] |
| inclusion_100.jpg | background | inclusion | 0.4111 | [np.float64(154.8), np.float64(178.4), np.float64(179.1), np.float64(200.0)] |
| inclusion_117.jpg | background | inclusion | 0.5936 | [np.float64(53.4), np.float64(148.1), np.float64(68.0), np.float64(192.3)] |
| inclusion_117.jpg | background | inclusion | 0.2766 | [np.float64(33.2), np.float64(153.1), np.float64(46.0), np.float64(191.9)] |
| inclusion_122.jpg | background | inclusion | 0.3713 | [np.float64(110.6), np.float64(140.7), np.float64(141.9), np.float64(200.0)] |
| inclusion_123.jpg | background | inclusion | 0.4895 | [np.float64(47.2), np.float64(127.3), np.float64(79.8), np.float64(200.0)] |
| inclusion_148.jpg | background | inclusion | 0.4497 | [np.float64(11.6), np.float64(0.1), np.float64(62.4), np.float64(87.8)] |
| inclusion_18.jpg | background | inclusion | 0.3444 | [np.float64(1.1), np.float64(0.0), np.float64(38.5), np.float64(90.0)] |
| inclusion_18.jpg | background | inclusion | 0.2816 | [np.float64(155.0), np.float64(104.7), np.float64(183.4), np.float64(198.2)] |
| crazing_109.jpg | crazing | background | N/A | [np.float64(77.0), np.float64(1.0), np.float64(199.0), np.float64(42.0)] |
| crazing_109.jpg | crazing | background | N/A | [np.float64(39.0), np.float64(38.0), np.float64(198.0), np.float64(87.0)] |
| crazing_109.jpg | crazing | background | N/A | [np.float64(32.0), np.float64(88.0), np.float64(145.0), np.float64(131.0)] |
| crazing_114.jpg | crazing | background | N/A | [np.float64(41.0), np.float64(139.0), np.float64(169.0), np.float64(188.0)] |
| crazing_114.jpg | crazing | background | N/A | [np.float64(5.0), np.float64(52.0), np.float64(46.0), np.float64(198.0)] |
