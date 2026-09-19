# VEHICLE_LEVEL_RECONSTRUCTION_AUDIT.md

## Status: **NOT EXECUTED** — blocked on raw dataset mount

**2026-09-19 continue:** Re-checked with `OCSLAB_DATASET_DIR` set to the user OneDrive path. Path still **not visible** on the Cloud Agent VM; **0 / 34** raw traces resolve (`RAW_TRACE_RESOLUTION.csv`). IF reconstruction remains blocked.

No Isolation Forest was re-fit in this checkpoint. Historical models and thresholds were **not** overwritten.

---

## Intended methodology (verified in code — not yet re-run)

| Step | Authoritative location | Rule |
|------|------------------------|------|
| Features | 24-D `BEHAVIOURAL_FEATURE_COLUMNS` | Benign-train fit for IF only |
| Windowing | size 100 / stride 50 | Within-segment only |
| IF fit | Benign **training** windows only | No attack labels in fit |
| Operating point | `src/evaluation/vehicle_level_evaluation.py` | Among validation thresholds with **FPR ≤ 5%**, choose **highest recall**; fallback FPR≤10% if none |
| Selected method label | `SELECTED_METHOD_LABEL = "FPR<=5%"` | Matches paper statement |
| Test labels | Evaluation only | Must not select thresholds |

**No discrepancy found** between the paper’s stated FPR≤5% + max-recall rule and the current `vehicle_level_evaluation.py` implementation. Reconstruction may proceed with that code once raw data exist.

### Note on fleet weak/strong thresholds

Balanced IF training manifest stores fleet-promotion style thresholds such as `{"weak": 0.45, "strong": 0.7}` (Chevrolet strong 0.85). The master YAML fleet local-IDS pair is weak **0.55** / strong **0.80**. These layers must not be conflated with the vehicle-level FPR≤5% score threshold used for table P4-style metrics. Exact reconciliation is deferred until IF re-scoring is possible.

---

## Historical reference checks (table P4 — immutable)

From `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv`:

| Scope | PR-AUC | Precision | Recall | F1 | Test FPR |
|-------|--------|-----------|--------|----|----------|
| Chevrolet | 0.994 | 0.894 | 0.998 | 0.943 | 0.152 |
| Hyundai | 0.826 | 0.715 | 0.669 | 0.691 | 0.364 |
| Kia | 0.985 | 0.950 | 0.891 | 0.920 | 0.306 |
| pooled | 0.969 | 0.911 | 0.862 | 0.886 | 0.310 |

User-cited approximate headlines (Precision 0.900 / Recall 0.838 / F1 0.868 / per-vehicle F1 0.959/0.798/0.883) are **close but not identical** to table_P4. For reconstruction audits, **table_P4 is the numeric reference**.

---

## Reconstruction classification

| Item | Classification |
|------|----------------|
| Newly reconstructed validation FPR/recall/thresholds | **NOT AVAILABLE** |
| Match vs historical | **N/A** (no run) |
| Proceed to η selection? | **No** |

When raw data are mounted, re-fit under the recovered split, write new models under this experiment tree only, and fill MATCH / CLOSE / MATERIAL_DIFFERENCE against table_P4 **without tuning**.
