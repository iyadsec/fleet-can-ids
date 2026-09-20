# S3_S4_FAILURE_ANALYSIS.md

**Runs analysed:** 20 (S3+S4 × 10 seeds)
**Missed GT campaigns (tp=0):** 19
**Matched (tp≥1):** 1

## Primary failure classification (missed runs only)

| Failure | Count | % of missed | % of all S3/S4 |
|---------|------:|------------:|---------------:|
| BETA_COHESION_FAILURE | 19 | 100.0% | 95.0% |

**ETA_GATE_FAILURE count:** 0 (consistent with 0/159 η=2 rejections on the full suite).

### Immediate mechanism

On **20/20** S3/S4 runs, ≥2 GT campaign vehicles co-occur in at least one pre-gate DBSCAN cluster.
Those GT-bearing clusters are typically **fleet-scale mega-clusters**:

| Stat (clusters with ≥2 GT vehicles) | S3 | S4 |
|-------------------------------------|----|----|
| count | 12 | 10 |
| mean \|C_k\| | 116.6 | 130.4 |
| mean r_k | 17.6 | 20.0 |
| mean c_k | 0.376 | 0.213 |
| fraction β pass (c_k≥0.5) | 0.17 | 0.00 |
| fraction η pass | 1.00 | 1.00 |
| mean J(P,G) | 0.298 | 0.250 |

γ passes whenever GT is co-clustered (r_k large). **β fails** because mega-cluster cohesion ≈0.2–0.4 ≪ 0.5.
η always passes on these clusters (size ≫ 2). Jaccard on mega-clusters is typically 0.25 (=5/20) < τ_J=0.5.

## Per-scenario failure breakdown

### S3
- BETA_COHESION_FAILURE: 9/10
- NONE_MATCHED: 1/10 (seed137: gate-passing cluster with J=0.71 → TP)

### S4
- BETA_COHESION_FAILURE: 10/10

## Where F1 collapses

- Runs with ≥2 GT vehicles co-clustered (pre-gate): **20/20**
- Runs with ≥1 gate-passing predicted campaign (γ∧β∧η): **2/20**
- Runs with campaign TP under Jaccard τ=0.5: **1/20**

**Collapse is AFTER clustering / AT the β cohesion gate**, not at η and not primarily at descriptor absence.

## Jaccard diagnostic (τ_J unchanged at 0.5)

### All pre-gate clusters vs GT

| Bucket | Count |
|--------|------:|
| J=0 | 35 |
| 0<J<0.25 | 3 |
| 0.25<=J<0.5 | 21 |
| 0.5<=J<0.75 | 1 |
| J>=0.75 | 0 |

### Gate-passing clusters only

| Bucket | Count |
|--------|------:|
| J=0 | 0 |
| 0<J<0.25 | 0 |
| 0.25<=J<0.5 | 1 |
| 0.5<=J<0.75 | 1 |
| J>=0.75 | 0 |

**DIAGNOSTIC ONLY:** S3/S4 runs with ≥1 gate-passing predicted campaign (irrespective of Jaccard): **2/20**

Runs with TP under τ_J=0.5: **1/20**

Conclusion: the revised matcher is **not** the dominant miss mode (only 2 gate-passing campaigns exist; 1 of them matches). Upstream β failure on over-merged clusters dominates.
