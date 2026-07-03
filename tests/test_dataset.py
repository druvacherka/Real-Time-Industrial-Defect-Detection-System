import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

dataset = PROJECT_ROOT / "datasets" / "raw" / "NEU-DET"

assert dataset.exists(), "Dataset not found."

print("Dataset found successfully.")

assert (dataset / "train").exists()

assert (dataset / "validation").exists()

print("Dataset structure verified.")