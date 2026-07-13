# Dataset Balancing & Augmentation Report

**Date/Time**: 2026-07-13 18:13:55
**Total Generation Time**: 1.61 seconds
**Target Instance Count**: 735

## 1. Class Distribution Before and After Augmentation

| Class ID | Class Name | Count Before | Count After | Generated | Change (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | crazing | 692 | 782 | 40 | +13.01% |
| 1 | inclusion | 720 | 763 | 15 | +5.97% |
| 2 | patches | 735 | 742 | 0 | +0.95% |
| 3 | pitted_surface | 692 | 741 | 39 | +7.08% |
| 4 | rolled-in_scale | 690 | 775 | 43 | +12.32% |
| 5 | scratches | 691 | 777 | 42 | +12.45% |

## 2. Augmentation Pipelines & Quality Checks
- Albumentations was configured with standard spatial and pixel transforms.
- Output JPEG quality was set to 95.
- Multi-threaded executor was utilized for disk read/write optimization.