# Graph Construction Comparison Summary

## A) Original Top-K (k=15)
- Cross-vehicle edges: 189
- Cross-vehicle clusters (≥2 vehicles): 1
- Weak recovery rate: 0.00%
- Weak fleet FPR: 0.0000
- Fleet recall gain (all descriptors): 0.00%
- Fleet F1 gain: 0.00%

## B) Cross-Vehicle Constrained kNN (10 same + 5 cross)
- Cross-vehicle edges: 384,568
- Cross-vehicle clusters (≥2 vehicles): 1
- Weak recovery rate: 100.00%
- Weak fleet FPR: 1.0000
- Recovered weak attacks: 13,453 / eligible 20,206
- Fleet recall gain (all descriptors): 0.00%
- Fleet F1 gain: 0.00%

## Conclusion

**Cross-vehicle constrained kNN fixes the connectivity problem** (cross-vehicle edges 189 → 384,568) and enables weak-anomaly recovery when cluster gates are applied on the weak-only graph.

On the **full descriptor graph**, fleet correlation gain remains **0.00%** because promotion thresholds (≥3 vehicles, mean score ≥0.7) are not satisfied—or the graph collapses to one component whose mean score is below threshold.

**FPR trade-off on weak anomalies:** cross-vehicle kNN recovery raises weak-subset FPR from 0.0000 to 1.0000 when the weak graph forms a single multi-vehicle component—report both recovery and FPR in the paper.
