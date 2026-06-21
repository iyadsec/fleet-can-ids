# Paper Publication Artifacts — Single Folder

**Location (use this folder only for paper writing):**

```
paper/publication_artifacts/
```

**Generated:** 2026-06-21 21:23 UTC  
**Total files:** 149

This folder consolidates **all latest figures and tables** from corrected CTT, cross-dataset, local IDS, fleet, and diagnostic outputs. Original experiment roots are unchanged on their source branches; these are copies for convenience.

## Subfolders

| Folder | What it contains |
|--------|------------------|
| `01_corrected_ctt/` | CTT_CORR1–7 tables & figures, corrected local/scenario results |
| `02_cross_dataset_ocslab_vs_ctt_corrected/` | Fair OCSLab vs corrected CTT comparison |
| `03_local_ids_ocslab_vs_ctt/` | LOCAL_COMP1–6 local IDS pooled/per-vehicle comparison |
| `04_fleet_level_corrected_summary/` | FLEET_CORR1–6 fleet scenario & consistency-rule summary |
| `05_ctt_f1_merge_diagnostics/` | Diagnostic figures explaining low F1 & unrelated merge |

Each subfolder has `figures/` and `tables/` (plus reports/validation/results as needed).

## Original scattered locations (before consolidation)

| Content | Original path |
|---------|---------------|
| Corrected CTT | `new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/` |
| Cross-dataset (corrected) | `new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/` |
| Local IDS comparison | `new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/` |
| Fleet summary | `new_experiments/fleet_level_corrected_comparison_summary/` |
| F1/merge diagnostics | `new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/` |

## Recommended for main paper

| Section | Table | Figure |
|---------|-------|--------|
| Local IDS | `03_local_ids_ocslab_vs_ctt/tables/LOCAL_COMP1_pooled_ocslab_vs_ctt.csv` | `03_local_ids_ocslab_vs_ctt/figures/figure_LOCAL_COMP1_pooled_comparison.png` |
| Fleet scenarios | `04_fleet_level_corrected_summary/tables/FLEET_CORR1_corrected_ctt_fleet_summary.csv` | `04_fleet_level_corrected_summary/figures/figure_FLEET_CORR2_unrelated_merge_before_after.png` |
| Cross-dataset | `02_cross_dataset_ocslab_vs_ctt_corrected/tables/table_CUR_COMP3_fleet_scenario_comparison.csv` | `02_cross_dataset_ocslab_vs_ctt_corrected/figures/figure_CUR_COMP3_scenario_outcomes.png` |

See `MANIFEST.csv` for full source traceability (git ref + original path per file).

## Regenerate this folder

```bash
python3 scripts/consolidate_paper_publication_artifacts.py
```
