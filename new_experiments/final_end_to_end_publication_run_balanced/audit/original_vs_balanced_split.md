# Original vs balanced split

## Chevrolet validation coverage

| Metric | Original (trace-level) | Balanced |
|--------|------------------------|----------|
| Benign validation traces | 0 | 4 segment(s) |
| Benign validation windows | within-trace only | 408 |

## Window counts by platform (balanced)

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

## Overlap checks

Balanced split validation: **PASS**.
Guard gap: **100 frames** (one full window length).

## Local IDS metrics (balanced run)

| vehicle_model   |   roc_auc |   pr_auc |   precision |   recall |       f1 |   false_positive_rate |   false_negative_rate |   latency_sec |   test_windows |
|:----------------|----------:|---------:|------------:|---------:|---------:|----------------------:|----------------------:|--------------:|---------------:|
| Chevrolet       |  0.993762 | 0.993597 |    0.893654 | 0.998084 | 0.942986 |              0.151961 |            0.00191571 |           0.1 |            930 |
| Hyundai         |  0.738374 | 0.825614 |    0.714894 | 0.668562 | 0.690952 |              0.363566 |            0.331438   |           0.1 |           3049 |
| Kia             |  0.90712  | 0.985398 |    0.949826 | 0.891075 | 0.919513 |              0.305791 |            0.108925   |           0.1 |          10615 |
| pooled          |  0.883706 | 0.968967 |    0.911218 | 0.861847 | 0.885845 |              0.30957  |            0.138153   |           0.1 |          14594 |

## Shared fleet configuration (balanced)

```json
{
  "similarity_threshold": 0.95,
  "max_same_vehicle_neighbors": 2,
  "max_cross_vehicle_neighbors": 5,
  "dbscan_eps": 0.5,
  "dbscan_min_samples": 2,
  "fragment_consolidation_enabled": true,
  "fragment_centroid_threshold": 0.85,
  "minimum_distinct_vehicles": 2,
  "minimum_cross_vehicle_support": 1,
  "minimum_campaign_cohesion": 0.5
}
```

## Scenario results comparison

| Metric | Original | Balanced |
|--------|----------|----------|
| Benign-Fleet Control (false_campaign_rate) | 0.000 | 0.000 |
| Strong campaign F1 (cs=2) | 0.600 | 0.533 |
| Weak campaign F1 (cs=2) | 0.100 | 0.067 |
| Strong campaign F1 (cs=5) | 0.867 | 0.733 |
| Weak campaign F1 (cs=5) | 0.533 | 0.500 |
| Strong campaign F1 (cs=10) | 0.933 | 1.000 |
| Weak campaign F1 (cs=10) | 0.450 | 0.717 |

### Shared fleet configuration (original)

```json
{
  "similarity_threshold": 0.95,
  "max_same_vehicle_neighbors": 2,
  "max_cross_vehicle_neighbors": 10,
  "dbscan_eps": 0.5,
  "dbscan_min_samples": 2,
  "fragment_consolidation_enabled": true,
  "fragment_centroid_threshold": 0.85,
  "minimum_distinct_vehicles": 2,
  "minimum_cross_vehicle_support": 1,
  "minimum_campaign_cohesion": 0.5
}
```

## Conclusion

Chevrolet validation coverage is restored via disjoint contiguous segments with guard gaps. Scenario-level metrics were regenerated; headline conclusions remain qualitatively similar when safety constraints hold, but Chevrolet validation participation may shift threshold selection and local Chevrolet scores relative to the original split.