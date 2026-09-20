# ETA_VALIDATION_AUDIT.md

**Verdict:** ETA_VALIDATION_COMPLETE

- Metric protocol: `revised_jaccard_0.5_v3` (FROZEN)
- Selected η: **2**
- η impact: **REDUNDANT_ON_VALIDATION**
- Candidates evaluated: [2, 3, 5, 10]
- Validation runs: 50 (S0–S4 × 10 seeds)
- DBSCAN labels identical across η: **Yes**
- Final test run: **No**
- CTT run: **No**
- Manuscript / June artifacts modified: **No**

## Selection summary

|   eta |   S0_false_campaign_rate |   S1_false_campaign_rate |   S2_incorrect_merge_rate |   S3_campaign_f1 |   S3_membership_f1 |   S4_campaign_f1 |   S4_membership_f1 |   selection_score |   clusters_rejected_by_eta |   pre_gate_clusters_total |   pct_clusters_rejected_by_eta | eligible   | selection_status   |
|------:|-------------------------:|-------------------------:|--------------------------:|-----------------:|-------------------:|-----------------:|-------------------:|------------------:|---------------------------:|--------------------------:|-------------------------------:|:-----------|:-------------------|
|     2 |                      0.3 |                      0.1 |                         0 |              0.1 |          0.0833333 |                0 |                  0 |              0.05 |                          0 |                       159 |                     0          | True       | SELECTED           |
|     3 |                      0.3 |                      0.1 |                         0 |              0.1 |          0.0833333 |                0 |                  0 |              0.05 |                          1 |                       159 |                     0.00628931 | True       | ELIGIBLE           |
|     5 |                      0.3 |                      0.1 |                         0 |              0.1 |          0.0833333 |                0 |                  0 |              0.05 |                          1 |                       159 |                     0.00628931 | True       | ELIGIBLE           |
|    10 |                      0.3 |                      0.1 |                         0 |              0.1 |          0.0833333 |                0 |                  0 |              0.05 |                          1 |                       159 |                     0.00628931 | True       | ELIGIBLE           |

## Mean ± std (validation seeds) — selected η

| Scenario | Campaign F1 | Membership F1 | False-campaign rate | Incorrect-merge rate | n_predicted |
|----------|-------------|---------------|---------------------|----------------------|-------------|
| S0 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3000 ± 0.4583 | n/a | 0.7000 ± 1.1000 |
| S1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1000 ± 0.3000 | n/a | 0.2000 ± 0.6000 |
| S2 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | n/a | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| S3 | 0.1000 ± 0.3000 | 0.0833 ± 0.2500 | n/a | n/a | 0.2000 ± 0.4000 |
| S4 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | n/a | n/a | 0.0000 ± 0.0000 |

## Per-η S3 / S4 Campaign F1 and selection_score

| eta | S3 CampF1 | S4 CampF1 | selection_score | status |
|-----|-----------|-----------|-----------------|--------|
| 2 | 0.1000 | 0.0000 | 0.0500 | SELECTED |
| 3 | 0.1000 | 0.0000 | 0.0500 | ELIGIBLE |
| 5 | 0.1000 | 0.0000 | 0.0500 | ELIGIBLE |
| 10 | 0.1000 | 0.0000 | 0.0500 | ELIGIBLE |

## Notes

- Predeclared tie rule `|Δ selection_score| ≤ 0.01` → prefer smaller η; all four candidates tied at 0.05 → η=2.
- Reconstructed suite campaign_size strata: S2/S3/S4 use size=5 only (2 and 10 absent).
- S2 GT incidents are reconstructed as five singleton-vehicle incidents; gate requires ≥2 vehicles, so campaign match under Jaccard τ=0.5 is structurally difficult — reported as measured.
- η=2 rejects 0 clusters that pass γ/β (all non-noise DBSCAN clusters already have size≥2); impact = REDUNDANT_ON_VALIDATION (calculated).

