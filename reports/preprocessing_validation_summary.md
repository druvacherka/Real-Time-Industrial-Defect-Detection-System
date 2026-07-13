# Preprocessing Pipeline Run Summary

**Execution Timestamp**: 2026-07-13 18:13:35
**Image Dimensions**: 640x640 (linear interpolation)
**Normalization Mode**: min_max (enabled=True)
**Output Format**: `.png`

## 1. Processed Split Counts

| Split | Total Input | Preprocessed Successfully | Skipped (Corrupt) |
| --- | --- | --- | --- |
| Train | 1930 | 1930 | 0 |
| Val | 360 | 360 | 0 |
| Test | 180 | 180 | 0 |

## 2. Performance Summary

- **Concurrency Level**: 4 Parallel Processes
- **Mean Processing Speed**: 42.03 ms per image
- **Total Time Spent**: 103.82 seconds

### Stage Timing Breakdown (Mean):
- **Read & Validate**: 1.57 ms
- **Resize**: 0.88 ms
- **Normalization**: 7.16 ms
- **Save & Format**: 32.41 ms

---
*Report generated automatically by `preprocess_pipeline.py`.*