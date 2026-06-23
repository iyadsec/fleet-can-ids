# Baseline and ablation comparison

Assembled from `origin/cursor/campaign-clustering` — **no experiments rerun**.

## Important

Read `RESULT_PROVENANCE.md` before citing numbers. M3 uses the **balanced publication** run;
M1/M2 use the **framework ablation** bundle (table_06).

## Reproduce

```bash
python scripts/build_baseline_ablation_comparison.py
python scripts/consolidate_experimental_results.py
```
