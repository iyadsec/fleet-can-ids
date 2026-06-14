# Balanced split validation

**Status:** PASS
**Guard gap (frames):** 100

## Platform summary

| vehicle_model   | split      |   source_trace_count |   source_segment_count |   benign_window_count |   malicious_window_count |   total_window_count |
|:----------------|:-----------|---------------------:|-----------------------:|----------------------:|-------------------------:|---------------------:|
| Hyundai         | train      |                    6 |                      1 |                  8377 |                    30725 |                39102 |
| Hyundai         | validation |                    1 |                      1 |                  1290 |                     5708 |                 6998 |
| Hyundai         | test       |                    1 |                      1 |                  1290 |                     1759 |                 3049 |
| Kia             | train      |                    6 |                      1 |                 10474 |                    20892 |                31366 |
| Kia             | validation |                    1 |                      1 |                  1416 |                    11325 |                12741 |
| Kia             | test       |                    1 |                      1 |                  1416 |                     9199 |                10615 |
| Chevrolet       | train      |                    0 |                      4 |                  1915 |                     2475 |                 4390 |
| Chevrolet       | validation |                    0 |                      4 |                   408 |                      522 |                  930 |
| Chevrolet       | test       |                    0 |                      4 |                   408 |                      522 |                  930 |

## Checks

1. Chevrolet/Hyundai/Kia benign train/validation/test windows.
2. No row/window in multiple splits.
3. No event ID duplicated across splits.
4. Guard gaps on segmented traces (100 frames).
5. Ready for downstream IF retraining.
6. Original E2E outputs isolated under separate root.
