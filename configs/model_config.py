"""
Model Configuration Manager
============================
Centralizes YOLOv8 model configuration, hyperparameter loading,
and experiment tracking setup for the defect detection project.

Author: druvacherka
Date: 2026-07-04
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from utils.config import ProjectConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Default training hyperparameters
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    "model": "yolov8n.pt",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "workers": 4,
    "device": "cpu",
    "patience": 50,
    "save_period": -1,
    "verbose": True,
}


def load_experiment_config(
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    Falls back to the default experiment.yaml in the configs directory.

    Args:
        config_path: Optional path to a custom config file.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = str(ProjectConfig.CONFIG_DIR / "experiment.yaml")

    path = Path(config_path)
    if not path.is_file():
        logger.warning("Config file not found: %s. Using defaults.", config_path)
        return dict(DEFAULT_HYPERPARAMS)

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    logger.info("Loaded experiment config from %s", config_path)
    return config


def merge_hyperparams(
    base: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge override hyperparameters into a base configuration.

    Args:
        base: Base hyperparameter dictionary.
        overrides: Override values to apply.

    Returns:
        Merged dictionary (base is not mutated).
    """
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged:
            logger.debug("Overriding %s: %s -> %s", key, merged[key], value)
        else:
            logger.debug("Adding new param %s: %s", key, value)
        merged[key] = value
    return merged


def build_training_args(
    data_yaml: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the final training argument dictionary for Ultralytics YOLO.

    Args:
        data_yaml: Path to the data.yaml file.
        overrides: Optional hyperparameter overrides.

    Returns:
        Complete training arguments dictionary.
    """
    args = dict(DEFAULT_HYPERPARAMS)
    args["data"] = data_yaml
    args["project"] = str(ProjectConfig.ROOT_DIR / "results")
    args["name"] = "defect_detection"
    args["exist_ok"] = True

    if overrides:
        args = merge_hyperparams(args, overrides)

    logger.info(
        "Training args built — epochs=%d, batch=%d, imgsz=%d, device=%s",
        args["epochs"],
        args["batch"],
        args["imgsz"],
        args["device"],
    )
    return args


def save_hyperparams(
    params: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Save hyperparameters to a YAML file for reproducibility.

    Args:
        params: Hyperparameter dictionary.
        output_path: Destination YAML file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(params, fh, default_flow_style=False, sort_keys=False)
    logger.info("Saved hyperparams to %s", output_path)


def get_model_variant(variant: str = "n") -> str:
    """
    Return the Ultralytics model identifier for a YOLOv8 variant.

    Args:
        variant: One of 'n', 's', 'm', 'l', 'x'.

    Returns:
        Model identifier string (e.g. 'yolov8n.pt').
    """
    valid_variants = {"n", "s", "m", "l", "x"}
    variant = variant.lower().strip()
    if variant not in valid_variants:
        logger.warning(
            "Unknown variant '%s', falling back to 'n'", variant
        )
        variant = "n"
    return f"yolov8{variant}.pt"
