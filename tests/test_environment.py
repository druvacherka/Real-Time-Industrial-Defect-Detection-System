import sys

print("=" * 60)
print("Industrial Defect Detection System")
print("Environment Verification")
print("=" * 60)

print(f"Python Version : {sys.version}")

try:
    import torch
    print(f"PyTorch        : {torch.__version__}")
except Exception as e:
    print("PyTorch Error:", e)

try:
    import cv2
    print(f"OpenCV         : {cv2.__version__}")
except Exception as e:
    print("OpenCV Error:", e)

try:
    import ultralytics
    print(f"Ultralytics    : {ultralytics.__version__}")
except Exception as e:
    print("Ultralytics Error:", e)

try:
    import numpy
    print(f"NumPy          : {numpy.__version__}")
except Exception as e:
    print("NumPy Error:", e)

try:
    import pandas
    print(f"Pandas         : {pandas.__version__}")
except Exception as e:
    print("Pandas Error:", e)

print("\nEnvironment verification completed successfully.")