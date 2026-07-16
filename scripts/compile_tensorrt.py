"""
TensorRT Engine Compilation Script.
===================================
Compiles the exported ONNX model into a hardware-accelerated TensorRT engine.
Supports FP32, FP16, and INT8 precision profiles.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import torch

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger("tensorrt_compilation")


def main():
    logger.info("Initializing TensorRT engine compilation configuration...")
    
    onnx_path = ProjectConfig.ROOT_DIR / "exports" / "model.onnx"
    engine_path = ProjectConfig.ROOT_DIR / "exports" / "model.engine"
    
    if not onnx_path.exists():
        logger.error(f"ONNX model source not found at: {onnx_path}")
        sys.exit(1)
        
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    
    # Check if trtexec exists in system PATH
    trtexec_path = shutil.which("trtexec")
    
    # Construct trtexec command
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",  # Enable FP16 compilation for edge performance
        "--workspace=3072",  # 3GB workspace memory limit
        "--avgRuns=10"
    ]
    
    logger.info(f"Target compilation command: {' '.join(cmd)}")
    
    if not gpu_available:
        logger.warning("CUDA GPU is not available in the current environment.")
        logger.info("Compilation cannot execute locally. Saving command configuration for target hardware.")
        
        # Save compilation script for the target device
        shrun_path = ProjectConfig.ROOT_DIR / "exports" / "compile_engine.sh"
        shrun_path.parent.mkdir(parents=True, exist_ok=True)
        shrun_path.write_text(f"#!/bin/bash\n{' '.join(cmd)}\n", encoding="utf-8")
        
        logger.info(f"Saved target compilation script to: {shrun_path}")
        logger.info("TensorRT compilation dry-run completed successfully.")
        return

    if trtexec_path is None:
        logger.error("trtexec tool was not found in system PATH. Ensure TensorRT is correctly installed.")
        sys.exit(1)
        
    try:
        logger.info("Starting TensorRT compilation via trtexec (this may take a few minutes)...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        logger.info("TensorRT compilation completed successfully.")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as err:
        logger.error(f"TensorRT compilation failed: {err.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
