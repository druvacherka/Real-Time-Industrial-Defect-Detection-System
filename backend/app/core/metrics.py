"""
Prometheus Metrics Instrumentation
==================================
Defines custom Prometheus metrics for tracking image preprocessing time,
model inference time, and total API prediction request latencies.
"""

try:
    from prometheus_client import Histogram, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

if PROMETHEUS_AVAILABLE:
    # Track time spent in image preprocessing
    PREPROCESSING_TIME = Histogram(
        "defect_detection_preprocessing_seconds",
        "Time spent in OpenCV preprocessing pipeline (seconds)",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
    )

    # Track time spent in YOLO inference
    INFERENCE_TIME = Histogram(
        "defect_detection_inference_seconds",
        "Time spent in YOLO model inference execution (seconds)",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5)
    )

    # Track total prediction latency
    TOTAL_LATENCY = Histogram(
        "defect_detection_prediction_latency_seconds",
        "Total time spent from request receipt to returning prediction (seconds)",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0)
    )

    # Track total prediction requests counter
    REQUEST_COUNTER = Counter(
        "defect_detection_requests_total",
        "Total count of defect detection requests processed",
        ["endpoint", "status"]
    )
else:
    # Dummy mock objects to prevent AttributeError if prometheus_client is absent
    class DummyMetric:
        def time(self):
            return self
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def observe(self, val):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self):
            pass

    PREPROCESSING_TIME = DummyMetric()
    INFERENCE_TIME = DummyMetric()
    TOTAL_LATENCY = DummyMetric()
    REQUEST_COUNTER = DummyMetric()
