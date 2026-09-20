# ETA_DECISION_IMPACT_VALIDATION.md

**Selected η = 2** (`revised_jaccard_0.5_v3`)

## Status: **REDUNDANT_ON_VALIDATION**

Computed as clusters that pass γ/β but fail `|C_k| ≥ η` (rejected_by_eta_only).
Total rejected for selected η = 0.
This is calculated from the gate, not inferred from DBSCAN `min_samples`.

## Totals (validation suite)

| Metric | Value |
|--------|-------|
| Pre-gate DBSCAN/post-fragment candidate clusters | 159 |
| Rejected because `|C_k| < η` only | 0 |
| Retained qualifying campaigns | 11 |
| % rejected by η | 0.00% |
| Campaign declarations changed by η vs η-free γ/β gate | 0 |

## Breakdown by scenario

| scenario   |   pre |   rej |   qual |   camp_f1 |   false_c |   merge |
|:-----------|------:|------:|-------:|----------:|----------:|--------:|
| S0         |    38 |     0 |      7 |       0   |       0.3 |       0 |
| S1         |    29 |     0 |      2 |       0   |       0   |       0 |
| S2         |    32 |     0 |      0 |       0   |       0   |       0 |
| S3         |    30 |     0 |      2 |       0.1 |       0   |       0 |
| S4         |    30 |     0 |      0 |       0   |       0   |       0 |

## Strong vs weak (selected η)

- S3 Campaign F1 = 0.1000
- S3 Membership F1 = 0.0833
- S4 Campaign F1 = 0.0000
- S4 Membership F1 = 0.0000

## Campaign size

Reconstructed validation scenarios use campaign_size=5 for S2/S3/S4
(sizes 2 and 10 not present in this suite).

## Notes

- DBSCAN / post-fragment labels verified identical across all η candidates.
- Fragment consolidation applied once with merge-eligibility at η=2
  (smallest candidate), then only the η gate varied.
- Final TEST and CTT were not run.
