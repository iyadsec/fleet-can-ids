# Weak Anomaly Recovery — Summary

## Anomaly counts
- **Total weak anomalies** (0.55 ≤ score < 0.8): 20,206
- **Total strong anomalies** (score ≥ 0.8): 57,027

## Cross-vehicle cluster evidence (Step 2–3)
- **Weak-anomaly graph nodes:** 20,206
- **Edges (top-k=15, sim ≥ 0.95):** 205,414
- **Graph density:** 0.00100628
- **Connected components:** 3
- **Average cluster size:** 6735.33
- **Cross-vehicle cluster rate:** 0.00%
- **Clusters with 1 / 2 / 3+ vehicles:** 3 / 0 / 0

## Recovery (Step 4–7)
- **Recovery eligible** (≥2 vehicles, size ≥2, mean score ≥0.55): 0
- **Actually recovered attack weak anomalies:** 0
- **Weak attack anomalies missed locally:** 13,453
- **Recovery rate:** 0.00%
- **Recall gain:** 0.00%
- **F1 gain:** 0.00%

## False-positive impact
- **Local FPR (weak subset):** 0.0000
- **Fleet FPR (weak subset):** 0.0000
- **FPR change:** +0.0000

## Conclusion

### Do weak anomalies form cross-vehicle clusters?
No meaningful cross-vehicle clusters were observed under the current graph construction.

**The experimental evidence does not support the hypothesis** on this dataset: weak anomalies did not form meaningful cross-vehicle clusters under the configured graph settings.

Parameters: weak_threshold=0.55, strong_threshold=0.8, top_k=15, similarity_threshold=0.95, minimum_vehicle_count=2, recovery_score_threshold=0.55.
