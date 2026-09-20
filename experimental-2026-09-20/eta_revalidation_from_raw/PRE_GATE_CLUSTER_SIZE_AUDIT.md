# PRE_GATE_CLUSTER_SIZE_AUDIT.md

## Full validation suite (from η sweep; all S0–S4)

- n_clusters = **159**
- min = 2.0, p25=2.0, median=2.0, p75=107.5, max=156.0
- mean=41.233, std=56.707

| Size bin | Count |
|----------|------:|
| size=2 | 95 |
| size=3 | 7 |
| size=4 | 0 |
| size=5 | 1 (the sole cluster in the 5–9 band; confirmed in S3/S4 detail) |
| size=6–9 | 0 |
| size≥10 | 56 |

**Minimum cluster size = 2.0** (DBSCAN `min_samples=2` implies all non-noise clusters have size ≥ 2).

## Why η=2 rejected zero clusters

Rejection requires pass γ/β and fail `|C_k| ≥ η`. With η=2 and min cluster size ≥ 2, every γ/β-passing cluster also passes η. Calculated rejected_by_eta_only for η=2 = **0**.

## Hypothetical rejections at higher η (from η sweep detail)

- η=2: rejected_by_eta_only=0/159 (0.00%), qualifying=11
- η=3: rejected_by_eta_only=1/159 (0.63%), qualifying=10
- η=5: rejected_by_eta_only=1/159 (0.63%), qualifying=10
- η=10: rejected_by_eta_only=1/159 (0.63%), qualifying=10

## Why metrics were identical across η despite some rejections

Runs where `n_qualifying` differed across η: **1**

- val_S0_seed163: n_qualifying by η = {2: 3, 3: 2, 5: 2, 10: 2}

Those rejected clusters (γ/β pass, size < higher η) did not change mean S3/S4 Campaign F1 because they were either false campaigns in non-scoring scenarios or did not create/remove a Jaccard≥0.5 match on S3/S4. Campaign metrics for S3/S4 remained flat across η.

## S3/S4-only pre-gate sizes (this diagnostic)

- n=60, min=2.0, median=2.0, max=156.0
- size=2: 35, size=3: 2, size=4: 0, size=5: 1, size=6–9: 0, size≥10: 22

