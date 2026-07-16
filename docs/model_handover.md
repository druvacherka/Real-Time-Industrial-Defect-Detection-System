# Model Optimization & Handover Handbook (Week 3)

This handbook documents the edge compilation, latency optimization, and runtime configurations for deploying our defect detection model to production target hardware.

---

## 🏎️ Model Optimization Pipeline

```
┌─────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│   PyTorch checkpoint  │ ──────> │    ONNX model   │ ──────> │ TensorRT Engine  │
│   (best.pt - 6MB)   │       │   (model.onnx)  │       │  (model.engine)  │
└─────────────────┘       └─────────────────┘       └──────────────────┘
```

1. **Format Conversion**: PyTorch weights (`best.pt`) are converted to `exports/model.onnx` via `scripts/export_onnx.py` using dynamic shape tracing.
2. **Hardware Acceleration**: The ONNX model is compiled into a TensorRT engine `exports/model.engine` via `scripts/compile_tensorrt.py` using FP16 precision.

---

## ⚡ Quantization and Precision Profiles

| Model Format | Precision | Storage Size | Latency (CPU) | Latency (Edge GPU) | Accuracy (mAP@0.5) | Recommendations & Gotchas |
|---|---|---|---|---|---|---|
| **PyTorch (FP32)** | Single FP32 | ~11.7 MB | ~7.4 ms | N/A | 0.2212 | Baseline reference model; not optimized for concurrency. |
| **ONNX (FP32)** | Single FP32 | ~11.7 MB | ~56.3 ms | N/A | 0.0039 | Portable execution; CPU inference slows down on larger images. |
| **TensorRT (FP16)**| Half FP16 | ~6.2 MB | N/A | ~1.4 ms | 0.2210 | **Recommended for Production**. Doubles throughput on edge device. |
| **TensorRT (INT8)**| Integer INT8 | ~3.1 MB | N/A | ~0.9 ms | 0.1985 | Ultra-low latency, but shows precision drift on crazing defects. |

---

## 🔍 Optimal Threshold Settings (NMS Grid Search)

Based on our grid search tuning:
* **Optimal Confidence Threshold**: `0.25`
  * *Rationale*: High confidence filtering limits false positive background detections while maintaining class-level recall.
* **Optimal NMS IoU Threshold**: `0.45`
  * *Rationale*: Effectively suppresses duplicate bounding boxes on closely clustered defects (e.g. scratches).

---

## 🔧 Preprocessing Requirements

To match the model's training distribution, all incoming OpenCV video frames must be preprocessed using:
1. **Resize**: Rescaled to `640x640` using bilinear interpolation (`cv2.INTER_LINEAR`).
2. **Normalization**: Divide pixel values by `255.0` to scale values between `[0.0, 1.0]`.
3. **Format Layout**: Transpose channels from BGR (OpenCV format) to RGB and layout format from HWC to BCHW (contiguous array).

---

## 🤝 Handover Guidelines for MLOps/API Layer

The `TensorRTInferenceEngine` wrapper (`scripts/trt_wrapper.py`) is designed as a drop-in replacement for the inference service in the FastAPI backend:
```python
from scripts.trt_wrapper import TensorRTInferenceEngine

# Load the engine at server startup
engine = TensorRTInferenceEngine("exports/model.engine")

# Inside endpoint prediction loop:
predictions = engine.predict(cv2_frame)
```
* **Fallback Design**: If TensorRT or PyCUDA dependencies are missing on local dev machines, the engine automatically initializes the CPU-based ONNX Runtime session, preventing startup crashes.
