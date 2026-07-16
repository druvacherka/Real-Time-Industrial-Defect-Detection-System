"""
TensorRT Inference Engine Wrapper.
==================================
Handles low-latency inference on the target edge device using TensorRT engine.
Automatically falls back to ONNX Runtime on CPU if GPU/TensorRT libraries are missing.
"""

import sys
import numpy as np
from pathlib import Path
import cv2

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("tensorrt_wrapper")


class TensorRTInferenceEngine:
    def __init__(self, engine_path: Path):
        self.engine_path = Path(engine_path)
        self.use_trt = False
        
        # Check TensorRT library and GPU availability
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            self.trt = trt
            self.cuda = cuda
            
            if self.engine_path.exists() and self.engine_path.suffix == ".engine":
                self.use_trt = True
                logger.info(f"Successfully loaded TensorRT libraries. Using engine: {self.engine_path}")
                self._load_trt_engine()
            else:
                logger.warning(f"TensorRT engine file not found at: {self.engine_path}. Falling back to ONNX.")
        except ImportError:
            logger.info("TensorRT or PyCUDA libraries not found. Falling back to ONNX Runtime backend.")
            
        if not self.use_trt:
            self._setup_onnx_fallback()

    def _load_trt_engine(self):
        """Loads and prepares the compiled TensorRT execution engine."""
        TRT_LOGGER = self.trt.Logger(self.trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, self.trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.Stream()
        
        # Allocate GPU memory buffers for inputs/outputs
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        
        for binding in self.engine:
            size = self.trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
            dtype = self.trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Host and device buffers
            host_mem = self.cuda.pagelocked_empty(size, dtype)
            cuda_mem = self.cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _setup_onnx_fallback(self):
        """Prepares ONNX Runtime inference session as a local fallback."""
        try:
            import onnxruntime as ort
            onnx_path = ProjectConfig.ROOT_DIR / "exports" / "model.onnx"
            if not onnx_path.exists():
                logger.error(f"Fallback ONNX model not found at: {onnx_path}")
                raise FileNotFoundError()
                
            self.ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            logger.info("ONNX Runtime CPU fallback session initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize ONNX Runtime fallback session: {e}")
            raise e

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses input image (BGR OpenCV frame) into normalized float32 tensor."""
        # Resize to YOLOv8 model input resolution (640x640)
        resized = cv2.resize(frame, (640, 640))
        
        # Normalize and transend dimensions to CHW format
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # CHW -> BCHW
        return np.ascontiguousarray(img)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Runs model prediction using either TensorRT engine or ONNX Runtime fallback."""
        input_data = self.preprocess(frame)
        
        if self.use_trt:
            # Transfer input data to GPU
            np.copyto(self.host_inputs[0], input_data.ravel())
            self.cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
            
            # Execute inference
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Transfer output results back to host memory
            for i in range(len(self.host_outputs)):
                self.cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
                
            self.stream.synchronize()
            return [out.reshape(self.engine.get_binding_shape(binding)) for binding, out in zip(self.engine[1:], self.host_outputs)]
        else:
            # Run ONNX Runtime CPU execution
            outputs = self.ort_session.run(None, {self.input_name: input_data})
            return outputs[0]


def main():
    logger.info("Running TensorRT wrapper mock validation...")
    engine_path = ProjectConfig.ROOT_DIR / "exports" / "model.engine"
    
    try:
        engine = TensorRTInferenceEngine(engine_path)
        
        # Simulate a mock image input frame (200x200 BGR)
        mock_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        outputs = engine.predict(mock_frame)
        
        logger.info("Mock inference executed successfully.")
        logger.info(f"Model outputs shape: {outputs.shape}")
        logger.info("TensorRT Inference Engine wrapper validated successfully.")
    except Exception as err:
        logger.error(f"Inference validation failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
