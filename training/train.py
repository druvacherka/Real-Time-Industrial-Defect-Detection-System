"""
YOLOv8 Model Training Script Skeleton.
"""

import argparse
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import ProjectConfig
from utils.device import get_device, print_device_info
from utils.logger import get_logger

logger = get_logger("training")


def load_config(config_path: Path) -> dict:
    """Loads configuration yaml file."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataset_config(data_yaml_path: Path) -> bool:
    """
    Validates that the YOLO data configuration file exists and contains the correct class names.
    """
    if not data_yaml_path.exists():
        logger.error(f"Data configuration YAML not found at: {data_yaml_path}")
        return False
        
    try:
        with open(data_yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Verify names/nc
        if "names" not in data:
            logger.error("Missing 'names' key in data.yaml")
            return False
            
        logger.info(f"Successfully validated data config. Number of classes: {len(data['names'])}")
        return True
    except Exception as e:
        logger.error(f"Error parsing data.yaml: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on NEU Metal Surface Defects dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ProjectConfig.CONFIG_DIR / "experiment.yaml"),
        help="Path to the experiment configuration YAML."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size."
    )
    parser.add_argument(
        "--img_size",
        type=int,
        help="Override image size."
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use ('cpu', 'cuda', etc.)."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override YOLOv8 model architecture (e.g. yolov8n, yolov8s)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (1 epoch, batch size 2, minimal workers)."
    )
    
    args = parser.parse_args()
    
    # 1. Load experiment settings
    config_path = Path(args.config)
    try:
        config = load_config(config_path)
        logger.info(f"Loaded config from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)
        
    # Get config variables with command line overrides
    epochs = args.epochs if args.epochs is not None else config["training"].get("epochs", 100)
    batch_size = args.batch_size if args.batch_size is not None else config["training"].get("batch_size", 16)
    img_size = args.img_size if args.img_size is not None else config["training"].get("image_size", 640)
    model_arch = args.model if args.model is not None else config["model"].get("architecture", "yolov8n")
    optimizer = config["training"].get("optimizer", "SGD")
    workers = config["training"].get("workers", 4)
    
    # 2. Determine device
    device_arg = args.device if args.device is not None else config["hardware"].get("device", "cpu")
    if device_arg == "auto":
        device = get_device()
    else:
        device = device_arg
        
    logger.info(f"Using device: {device}")
    
    # Setup results output directory
    results_dir = ProjectConfig.ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Results output directory resolved: {results_dir}")
    
    # Dry-run override (1 epoch, batch 2, small size)
    if args.dry_run:
        logger.info("Dry-run mode activated: Overriding hyperparameters for quick validation.")
        epochs = 1
        batch_size = 2
        img_size = 64
        workers = 0
        optimizer = "Adam"
        
    # Log hyperparameters
    logger.info("--- Hyperparameters ---")
    logger.info(f"Model Architecture : {model_arch}")
    logger.info(f"Epochs             : {epochs}")
    logger.info(f"Batch Size         : {batch_size}")
    logger.info(f"Image Size         : {img_size}")
    logger.info(f"Optimizer          : {optimizer}")
    logger.info(f"Workers            : {workers}")
    logger.info("-----------------------")
    
    # 3. Initialize YOLOv8 Model
    model_name = f"{model_arch}.pt"
    logger.info(f"Initializing YOLO model: {model_name}")
    try:
        # Load a pretrained model if set, otherwise initialize a new model
        model = YOLO(model_name)
        logger.info("YOLO model initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing YOLO model: {e}")
        sys.exit(1)
        
    # 4. Model Training Data Config Verification
    logger.info("Preparing data configuration mapping...")
    data_yaml_path = ProjectConfig.CONFIG_DIR / "data.yaml"
    
    if not validate_dataset_config(data_yaml_path):
        logger.error("Dataset validation failed. Please ensure datasets are preprocessed and configs/data.yaml is present.")
        sys.exit(1)
        
    # 5. Invoke YOLOv8 Model Training Loop
    logger.info("Starting YOLOv8 training execution...")
    project_name = config["project"].get("name", "Real-Time Industrial Defect Detection")
    run_name = f"train_{model_arch}"
    
    try:
        # Run training
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(results_dir),
            name=run_name,
            optimizer=optimizer,
            workers=workers,
            exist_ok=True
        )
        logger.info("YOLOv8 training completed successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)
        
    params = {
        "model_arch": model_arch,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "device": str(device),
        "data_yaml": str(data_yaml_path),
        "results_dir": str(results_dir),
        "run_name": run_name
    }
    return params


if __name__ == "__main__":
    main()
