# Final Weak Anomaly Recovery — Summary

## Research questions

### 1. Can fleet correlation recover weak anomalies?
**Yes, partially.** Under DBSCAN sub-cluster promotion on the cross-vehicle constrained graph, the sweep recovered up to **1.41%** of locally missed weak attacks (maximum over 64,512 configurations). Original Top-K weak-only recovery: **0.00%**; Cross-Vehicle kNN weak-only: **100.00%** (the latter promotes all weak nodes in one connected component — FPR=1.0000).

### 2. Maximum achievable recovery?
- **1.41%** at DBSCAN eps=1.5, min_samples=5, mean score ≥ 0.55, strong support ≥ 20% (FPR=0.0754, F1=0.0269).

### 3. Best balanced operating point?
- **Balanced** (FPR ≤ 10%): recovery 1.41%, F1=0.0269, FPR=0.0754 (constraint satisfied).

### 4. Recommended deployment operating point?
- **Paper operating point: IEEE Recommended** — recovery ≥ 10% and FPR ≤ 10%, maximize F1. **IEEE constraints are infeasible** on this dataset (no config with recovery ≥ 10% and FPR ≤ 10%). **Deployment fallback: Balanced operating point** — recovery 1.41%, F1=0.0269, FPR=0.0754 (eps=1.5, min_samples=5, mean ≥ 0.55, cohesion ≥ 0.90, strong ≥ 20%).

### 5. Trade-off between recovery and false positives?
Higher recovery generally increases FPR. Across 64,512 configurations, recovery spans [0.00%, 1.41%] and FPR spans [0.0000, 0.0754]. See `figures/recovery_vs_fpr_curve.pdf` and `figures/weak_recovery_pareto_frontier.pdf`.

## Grid search
- DBSCAN (eps × min_samples): 16 × 4 = 64 clusterings
- Promotion configs per clustering: 1,008
- Total evaluated: 64,512

## DBSCAN eps scale
DBSCAN runs in scaled 8-D PCA space (Euclidean distance). Values ≤ 0.20 assign all nodes to noise; operational clusters appear at eps ≥ 0.90 (see `selective_weak_promotion`, eps=1.2).

## Graph construction
- Cross-vehicle constrained kNN (10 same + 5 cross, τ=0.95)
- Clustering: DBSCAN on all anomaly descriptors (weak + strong)
