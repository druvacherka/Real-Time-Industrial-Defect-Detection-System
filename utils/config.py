"""
Project configuration.
"""

from pathlib import Path


class ProjectConfig:

    ROOT_DIR = Path(__file__).resolve().parent.parent

    DATASET_DIR = ROOT_DIR / "datasets"

    RAW_DATASET = DATASET_DIR / "raw"

    PROCESSED_DATASET = DATASET_DIR / "processed"

    CONFIG_DIR = ROOT_DIR / "configs"

    TRAINING_DIR = ROOT_DIR / "training"

    CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

    LOG_DIR = TRAINING_DIR / "logs"

    EXPERIMENT_DIR = TRAINING_DIR / "experiments"

    WEIGHTS_DIR = TRAINING_DIR / "weights"