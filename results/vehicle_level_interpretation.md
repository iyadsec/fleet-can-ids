# Vehicle-Level IDS — Threshold Interpretation

## Why not use the F1-optimal threshold?

On the held-out test split, the **F1-optimal** validation threshold (0.059621) yields recall 98.53% but a false positive rate of **94.20%**. That means most benign CAN windows would be flagged as attacks, which is impractical for in-vehicle deployment.

## Why use an FPR-constrained threshold?

Automotive CAN intrusion detection must limit false alarms so operators can trust alerts. We therefore select the threshold on **validation data only**, requiring validation FPR ≤ 5%, then choosing the threshold with the **highest validation recall** among feasible candidates.

## Selected paper result

- **Selected method:** FPR<=5%
- **Threshold value:** 0.955825
- **Test recall:** 45.97%
- **Test precision:** 97.24%
- **Test F1:** 62.43%
- **Test FPR:** 4.02%
- **Test detection latency:** 74.82 ms

## Trade-off

Compared with the F1-optimal threshold, the FPR-controlled rule reduces false positives at the cost of lower attack recall. This is expected: stricter control of benign false alarms necessarily misses some subtle attack windows that score below the higher threshold.
