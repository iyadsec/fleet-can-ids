# REPRODUCTION_CHECK.md

## Goal

Determine whether a recovered metric/gate implementation reproduces:

- Strong Campaign F1: `0.533 / 0.733 / 1.000`
- Weak Campaign F1: `0.067 / 0.500 / 0.717`

without overwriting publication outputs.

Temporary verification directory: `outputs/publication_metric_recovery/verification/`.

---

## What was run

Script: `outputs/publication_metric_recovery/verify_artifact_identities.py`  
Result JSON: `verification/reproduction_check.json` → **`status: pass`**

### Check A — Aggregate reproduction (no new executor)

Mean of `results/campaign_metrics.csv` grouped by `campaign_size` for `strong_campaign` / `weak_campaign` **bit-matches** `tables/table_P7_*.csv` / `table_P8_*.csv` for the metric columns including F1.

| Table | F1 values |
|-------|-----------|
| P7 strong | 0.533333…, 0.733333…, 1.0 |
| P8 weak | 0.066666…, 0.5, 0.716666… |

**Grade:** `VERIFIED_FROM_PUBLICATION_ARTIFACT` / `VERIFIED_FROM_EXECUTED_CODE` (verification script only).

This proves P7/P8 are seed-means of the saved run-level metrics. It does **not** prove we recovered the code that created those run-level rows.

### Check B — Algebraic identities on saved rows

All identities listed in `METRIC_DEFINITION_AUDIT.md` / `CAMPAIGN_GATE_AUDIT.md` hold on the 60 strong+weak rows (see JSON).  

**Grade:** `VERIFIED_FROM_PUBLICATION_ARTIFACT`.

### Check C — Full primary experiment reproduction

**Not attempted / not possible from git alone.**

Missing blockers (non-exhaustive):

- `run_refinement_fcgnn`
- `extract_run_metrics`, `safety_row`
- `SharedFleetConfiguration`, `joint_parameter_search`, `build_mixed_validation_suite`
- `_run_single_test`, `PUBLICATION_SCENARIOS`, base `generate_tables`
- Balanced descriptors / IF models / `processed/window_features.csv`
- Per-run dumps under `results/scenario_evaluation/runs/`

**Grade:** `UNRECOVERABLE` for executable reproduction of field computation.

Inventing a replacement `extract_run_metrics` from the algebraic identities and re-running was **explicitly not done** (would violate recovery rules).

---

## Strongest evidence ladder

| Evidence | Status |
|----------|--------|
| Frozen P7/P8 numbers match summary | Yes |
| P7/P8 = mean(campaign_metrics) | Yes |
| Algebraic identities on campaign_metrics | Yes |
| Recovered executed emitter reproduces fields from raw clusters | **No** |
| End-to-end re-run matches P7/P8 from descriptors | **No** |

---

## Conclusion

Partial verification of **artifact consistency** succeeded.  
**Executable recovery / true reproduction of the metric implementation did not.**
