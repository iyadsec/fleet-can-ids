# VALIDATION_SCENARIO_AUDIT.md

## Status: **NOT CONSTRUCTED** — blocked on validation descriptors

**2026-09-19 continue:** Still blocked — no validation descriptors (raw traces unresolved). No S0–S4 scenarios materialised.

---

## Historical validation scenario metadata (reference)

`recovered_publication_pipeline/.../validation_scenarios/validation_manifest.csv` documents the **historical** validation suite used by missing `build_mixed_validation_suite` / `joint_parameter_search`:

| Scenario | Role (from historical naming) | Rows in manifest |
|----------|--------------------------------|------------------|
| V0 | Benign fleet control | 10 |
| V1 | Isolated / coordinated-rate probe | 10 |
| V2 | Incorrect-merging / multi-incident | 10 |
| V3 | Strong campaign (F1 in search objective) | 19 |
| V4 | Weak campaign | 19 |

Seeds: `131,137,149,157,163,179,181,191,193,197` (and additional V3/V4 seeds).  
`overlap_with_test` recorded as **0** where present.

Most historical `event_ids` cells are **empty** in the committed CSV — the manifest alone cannot rebuild scenario node sets.

---

## Publication fleet geometry (to reuse when constructing)

From balanced master / P7 artifacts:

| Parameter | Value |
|-----------|-------|
| Fleet size | 20 vehicles |
| Descriptors / vehicle | 10 |
| Total nodes | 200 |
| Campaign sizes (test tables) | 2 / 5 / 10 |

Validation search diagnostics (`selection_report.md`) used mean V3/V4 campaign F1 under the frozen graph/DBSCAN/cohesion config.

---

## Feasibility once raw data exist

| Requirement | Status |
|-------------|--------|
| Validation windows in split | **Yes** (20,669) |
| Enough validation attack diversity for S3/S4 | **Likely** (Hyundai malfunction val; Kia replay val; Chevrolet flood/fuzzy/malfunction val segments) |
| Builder module `build_mixed_validation_suite` | **Missing** — must reimplement carefully from peer `model_diversity_final_tuned/validation_scenarios.py` + publication budgets **without** inventing undocumented rules |
| Synthetic CAN frames | **Forbidden** |

If a scenario cannot be built without test leakage or synthetic frames, that scenario must be reported and skipped — not fabricated.

---

## Paper S0–S4 vs historical V0–V4

Treat S0–S4 as the paper’s controlled-scenario names and map carefully to V0–V4 / PUBLICATION_SCENARIOS once the missing registry is recovered or explicitly redefined **before** η evaluation. Do not silently rename metrics.
