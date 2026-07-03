import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig

print("=" * 50)

print("Project Configuration")

print("=" * 50)

print(ProjectConfig.ROOT_DIR)

print(ProjectConfig.DATASET_DIR)

print(ProjectConfig.LOG_DIR)

print(ProjectConfig.CHECKPOINT_DIR)