# Development Guide

## Utility Modules

### device.py

Detects CPU/GPU and prints device information.

### logger.py

Creates a reusable logger for all project modules.

### config.py

Stores project directory paths.

### constants.py

Stores project-wide constants such as class names and dataset information.

---

## Tests

Run:

```bash
python tests/test_device.py
python tests/test_logger.py
python tests/test_config.py
```

before starting development.

---

## Dataset Verification

Before starting model training, run:

```bash
python scripts/verify_dataset.py
python scripts/dataset_statistics.py
python tests/test_dataset.py
```

These scripts ensure:

- Dataset exists
- Folder structure is valid
- Annotation files are present
- Class distribution can be inspected