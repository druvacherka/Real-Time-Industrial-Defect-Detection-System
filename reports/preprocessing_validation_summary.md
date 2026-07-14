# Preprocessing Pipeline Run Summary

**Execution Timestamp**: 2026-07-14 21:49:53
**Image Dimensions**: 640x640 (linear interpolation)
**Normalization Mode**: min_max (enabled=True)
**Output Format**: `.png`

## 1. Processed Split Counts

| Split | Total Input | Preprocessed Successfully | Skipped (Corrupt) |
| --- | --- | --- | --- |
| Train | 2109 | 2109 | 0 |
| Val | 360 | 360 | 0 |
| Test | 180 | 180 | 0 |

## 2. Performance Summary

- **Concurrency Level**: 4 Parallel Processes
- **Mean Processing Speed**: 84.53 ms per image
- **Total Time Spent**: 223.92 seconds

### Stage Timing Breakdown (Mean):
- **Read & Validate**: 6.1 ms
- **Resize**: 1.27 ms
- **Normalization**: 12.76 ms
- **Save & Format**: 51.3 ms

---
*Report generated automatically by `preprocess_pipeline.py`.*