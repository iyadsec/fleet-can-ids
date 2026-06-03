# Selective Weak Anomaly Promotion — Summary

## Promotion gates (fixed)
- **Cluster unit:** DBSCAN sub-clusters on cross-vehicle graph nodes (eps=1.2, min_samples=10)
- Cluster vehicles ≥ 2
- Cluster size ≥ 5
- Mean pairwise similarity (cohesion) ≥ 0.97
- Mean anomaly score threshold (swept): [0.6, 0.7, 0.8]
- Strong anomaly support (swept): ['40%', '60%', '80%']

## Recommended operating point
- **Mean anomaly score ≥ 0.60**
- **Strong support ≥ 40%**
- **Recovery rate:** 0.07%
- **Recall:** 0.0007 (local baseline 0.0000)
- **Precision:** 0.0789
- **F1:** 0.0013 (local baseline 0.0000)
- **FPR:** 0.0155
- **Eligible clusters:** 4
- **Promoted weak windows:** 114
- **Recovered weak attacks:** 9

## Interpretation

DBSCAN produced **10** sub-clusters (67358 noise nodes). Promotion gates are applied per sub-cluster, not the single connected component.

The fleet graph forms **one connected component**, but DBSCAN sub-clusters enable selective promotion. Best F1=0.0013 at mean score ≥ 0.60, strong support ≥ 40% (recovery 0.07%, FPR 0.0155).

**Precision-safe alternative:** strong support ≥ 80% → 0 promotions.



Graph: cross-vehicle constrained kNN (10 same + 5 cross); clustering: DBSCAN on behavioural descriptors.
