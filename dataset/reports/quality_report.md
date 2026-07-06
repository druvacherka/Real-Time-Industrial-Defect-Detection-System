# Dataset Quality & Preprocessing Report

## 1. Summary Statistics

| Split | Images | Labels | Total Bounding Boxes |
| --- | --- | --- | --- |
| Train | 1930 | 1930 | 4220 |
| Val | 360 | 360 | 832 |
| Test | 180 | 180 | 441 |

## 2. Bounding Box Class Distribution

| Class Name | Train (incl. Aug) | Validation | Test |
| --- | --- | --- | --- |
| crazing | 692 | 129 | 65 |
| inclusion | 720 | 204 | 117 |
| patches | 735 | 184 | 91 |
| pitted_surface | 692 | 92 | 47 |
| rolled-in_scale | 690 | 118 | 72 |
| scratches | 691 | 105 | 49 |

## 3. Data Augmentation & Balancing

To handle class imbalance (the original NEU dataset has equal image splits but unequal bounding box instances), we implemented offline Albumentations augmentation targeting minority classes in the training split. Bounding box instances for all classes were augmented to match the majority class size (~690 instances).

The following Albumentations pipelines were configured:
- **Spatial Transforms**: Horizontal Flip, Vertical Flip, Safe Rotation
- **Pixel Transforms**: Random Brightness Contrast, Hue Saturation Value, CLAHE
- **Blur & Noise**: Gaussian Blur, Motion Blur

## 4. Integrity Verification
- **Missing Files**: 0 missing images/annotations detected.
- **Corrupt Files**: 0 corrupt images/annotations detected.
- **Duplicate Images**: Detected and handled duplicate patches set (`patches_101.jpg` / `patches_105.jpg`).
- **Image-Label Alignment**: Checked and confirmed that 100% of images have corresponding YOLO `.txt` format labels across all splits.
