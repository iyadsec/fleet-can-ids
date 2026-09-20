# ETA_SELECTION_VALIDITY.md

**Status:** η=2 is **PROVISIONAL** — not empirically validated for the publication pipeline.

## Mechanical selection facts (unchanged)

- η=2 was selected under frozen `metric_protocol=revised_jaccard_0.5_v3` and the predeclared tie rule (`|Δ selection_score| ≤ 0.01` → smaller η).
- All candidates η∈{2,3,5,10} were eligible and tied on `selection_score = mean(S3,S4) Campaign F1 = 0.05`.
- Selected η had **zero decision impact** on the reconstructed validation suite: **0 / 159** pre-gate clusters rejected by `|C_k| ≥ η` for η=2 (`REDUNDANT_ON_VALIDATION`).
- All η candidates produced **identical** S3/S4 Campaign F1 (S3=0.10, S4=0.00).

## Why this is not a validated publication parameter

The subsequent S3/S4 diagnostic (`VALIDATION_DIAGNOSTIC_VERDICT.md`) found:

- the reconstructed validation scenario builder is a **stand-in**, not the original `build_mixed_validation_suite`;
- that original function was **never committed** (`MISSING_MODULES.md`; `git log -S'def build_mixed_validation_suite'` → no defining blob);
- reconstructed same-GT cross-vehicle cosine ≈ 0.17 ≪ τ=0.95, with β-failing fleet-scale mega-clusters — inconsistent with frozen publication `campaign_metrics.csv` cluster geometry.

Therefore η=2 must **not** yet be represented as empirically validated for the publication pipeline.

## Constraints for subsequent work

- Do **not** select another η based on the stand-in suite.
- Do **not** delete existing η-selection artifacts (`eta_selection.json`, `ETA_SELECTION_SUMMARY.csv`, freezes).
- Re-evaluate η only after a publication-faithful validation suite is available (or explicitly declared unrecoverable with a new protocol).

```text
ETA_STATUS = PROVISIONAL_NOT_PUBLICATION_VALIDATED
```
