# Threshold Recommendation (Diagnostic)

**DIAGNOSTIC ONLY — official CTT3 tables unchanged.**

## Recommended evaluation ground truth

For fair local detection reporting on CTT, use:

`eval_label = (label==1) OR (attack_type != 'benign')` for metrics only.

Never use attack_type as model input.

## Why official pooled F1 is ~4.2%

CTT3 uses **strong mode** (97.5th percentile benign threshold) with **label-only** ground truth, then **means across four sets**. Only set_01 Impala has label=1 attack windows; other known vehicles (Silverado, etc.) contribute recall=0, precision≈0 → pooled artefact.

## Policy comparison (known vehicle, eval ground truth)

| Policy | Description | Diagnostic outcome |
|--------|-------------|-------------------|
| A | Existing strong threshold | Low F1; high recall on label=1 only (set_01); misses label=0 attacks |
| B | F1-optimal on scores | Best F1 under eval ground truth; per-set thresholds |
| D | FPR ≤ 5% | Aligns with OCSLab calibration target; moderate F1 gain |
| G | Recall ≥ 80% | Raises recall on eval positives; lower precision |

See `threshold_policy_comparison.csv` and `threshold_sweep_summary.csv` for full curves.

## Recommendation

1. Adopt **eval ground truth** for CTT local reporting (diagnostic until source labels are corrected).
2. Use **FPR≤5%** threshold policy for OCSLab-comparable operating point, or **F1-optimal** under eval labels for descriptive cross-dataset tables.
3. Do **not** overwrite official CTT3 until ground-truth policy is approved.
