"""
ONNX runtime wrapper wrapper for optimized defect detection inference.
Author: prajwaledu802-coder
"""
import numpy as np

class ONNXModelWrapper:
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.session = None

    def initialize_session(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.onnx_path)
        except ImportError:
            pass

    def predict(self, processed_image: np.ndarray):
        if not self.session:
            return []
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: processed_image})
