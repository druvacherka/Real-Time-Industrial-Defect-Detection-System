import logging
from pathlib import Path

from utils.config import ProjectConfig


def get_logger(name: str = "industrial_defect"):

    ProjectConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        ProjectConfig.LOG_DIR / "training.log"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger