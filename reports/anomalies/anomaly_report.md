# Dataset Anomaly Detection Report

Generated dynamically by `detect_anomalies.py`.

## 📊 Summary Statistics
- **Total Images Scanned**: 2649
- **Total Labels Scanned**: 2649
- **Corrupted Images**: 0
- **Image Anomalies Detected**: 1544
- **Label/Annotation Anomalies Detected**: 0

## 🖼️ Anomalous Images Details (Top 20)
| Split | Filename | Anomalies | Stats (Blur / Brightness / Contrast / Entropy) |
| --- | --- | --- | --- |
| train | aug_aug_crazing_13_61_00011.jpg | Image too bright (average: 250.73), Low contrast (std: 12.70), Low information entropy (entropy: 2.39) | Blur: 795.7, Bright: 250.7, Contrast: 12.7, Entropy: 2.39 |
| train | aug_aug_crazing_171_35_00039.jpg | Blurry image (variance: 8.98), Image too dark (average: 2.56), Low contrast (std: 6.15), Low information entropy (entropy: 2.07) | Blur: 9.0, Bright: 2.6, Contrast: 6.1, Entropy: 2.07 |
| train | aug_aug_pitted_surface_124_263_00014.jpg | Blurry image (variance: 56.74) | Blur: 56.7, Bright: 154.0, Contrast: 69.9, Entropy: 6.57 |
| train | aug_aug_pitted_surface_178_83_00004.jpg | Blurry image (variance: 5.69) | Blur: 5.7, Bright: 114.7, Contrast: 43.2, Entropy: 6.47 |
| train | aug_aug_pitted_surface_204_249_00029.jpg | Blurry image (variance: 41.33) | Blur: 41.3, Bright: 205.7, Contrast: 85.1, Entropy: 4.57 |
| train | aug_aug_pitted_surface_218_154_00020.jpg | Blurry image (variance: 3.84), Image too dark (average: 25.34) | Blur: 3.8, Bright: 25.3, Contrast: 17.6, Entropy: 5.52 |
| train | aug_aug_pitted_surface_228_267_00009.jpg | Blurry image (variance: 11.04) | Blur: 11.0, Bright: 146.8, Contrast: 64.9, Entropy: 6.90 |
| train | aug_aug_pitted_surface_228_85_00042.jpg | Blurry image (variance: 8.55) | Blur: 8.6, Bright: 173.4, Contrast: 34.6, Entropy: 6.93 |
| train | aug_aug_pitted_surface_253_17_00040.jpg | Blurry image (variance: 21.09) | Blur: 21.1, Bright: 141.0, Contrast: 24.6, Entropy: 6.39 |
| train | aug_aug_pitted_surface_277_209_00003.jpg | Blurry image (variance: 24.33) | Blur: 24.3, Bright: 198.9, Contrast: 73.2, Entropy: 6.06 |
| train | aug_aug_pitted_surface_281_170_00005.jpg | Blurry image (variance: 42.29) | Blur: 42.3, Bright: 174.8, Contrast: 65.7, Entropy: 6.85 |
| train | aug_aug_rolled-in_scale_162_15_00026.jpg | Blurry image (variance: 23.60) | Blur: 23.6, Bright: 122.2, Contrast: 59.5, Entropy: 6.60 |
| train | aug_aug_rolled-in_scale_177_117_00033.jpg | Blurry image (variance: 88.34) | Blur: 88.3, Bright: 176.0, Contrast: 75.4, Entropy: 6.58 |
| train | aug_aug_rolled-in_scale_233_74_00027.jpg | Blurry image (variance: 77.76) | Blur: 77.8, Bright: 59.0, Contrast: 29.6, Entropy: 5.19 |
| train | aug_aug_rolled-in_scale_261_69_00009.jpg | Blurry image (variance: 79.24), Image too dark (average: 16.32), Low contrast (std: 14.52) | Blur: 79.2, Bright: 16.3, Contrast: 14.5, Entropy: 3.88 |
| train | aug_aug_rolled-in_scale_66_80_00004.jpg | Blurry image (variance: 18.77) | Blur: 18.8, Bright: 93.5, Contrast: 48.0, Entropy: 5.83 |
| train | aug_aug_scratches_100_110_00026.jpg | Blurry image (variance: 8.64) | Blur: 8.6, Bright: 97.5, Contrast: 51.1, Entropy: 6.40 |
| train | aug_aug_scratches_100_115_00007.jpg | Blurry image (variance: 52.76) | Blur: 52.8, Bright: 91.0, Contrast: 48.0, Entropy: 6.63 |
| train | aug_aug_scratches_152_45_00035.jpg | Blurry image (variance: 13.11), Image too dark (average: 29.97) | Blur: 13.1, Bright: 30.0, Contrast: 28.5, Entropy: 5.68 |
| train | aug_aug_scratches_164_61_00041.jpg | Blurry image (variance: 33.08) | Blur: 33.1, Bright: 73.9, Contrast: 22.2, Entropy: 5.87 |

## 🏷️ Anomalous Labels Details (Top 20)
| Split | Filename | Anomalies |
| --- | --- | --- |
