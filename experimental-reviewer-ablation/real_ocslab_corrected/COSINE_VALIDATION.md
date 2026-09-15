# Cosine Validation — BLOCKED

Seed-11 cosine validation was **not executed**.

## Why

Graph cosine validation requires:

1. Manuscript-aligned Hyundai / Kia / Chevrolet **TEST** windows
2. Isolation Forest anomaly scores on those windows
3. Publication fleet benign z-score applied to 9-D `g_i` **before** cosine

Step 1 fails: `Dataset/ocslab_pipeline/Hyundai` and `Kia` are empty; `Chevrolet` holds classic Car-Hacking files, not Spark car_track CSVs (`DATASET_PROVENANCE.md`).

## Planned validity check (when data arrives)

For strong and weak seed-11 scenarios, after `g_i → fleet_benign_scaler → cosine`:

| Bucket | Stats |
|--------|-------|
| campaign–campaign | mean, median, min, max, % ≥ 0.95 |
| campaign–benign | same |
| campaign–unrelated | same |

**Hard gate:** if essentially 100% of all pair buckets still exceed 0.95 (PR #22 collapse), STOP before M1–M4.

## Scaler pre-check (already verified)

`fleet_benign_scaler.json` fitted feature names **exactly match** ablation `FEATURE_NAMES` (9-D), order-preserving. Safe to apply when features exist; must **not** fit on TEST.
