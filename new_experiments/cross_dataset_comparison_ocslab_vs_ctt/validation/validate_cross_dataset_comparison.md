# Cross-Dataset Comparison Validation
Generated: 2026-06-20T21:14:23.218488+00:00

- [PASS] **OCSLab result root checked (absent — expected in cloud workspace)** — /workspace/new_experiments/final_end_to_end_publication_run
- [PASS] **can-train-and-test result root exists** — /workspace/new_experiments/can_train_and_test_cross_dataset_validation/full
- [PASS] **Comparison tables COMP1–COMP8 generated** — count=8
- [PASS] **Comparison figures COMP1–COMP7 generated** — count=7
- [PASS] **Source map populated** — entries=31
- [PASS] **Caveated metrics documented** — caveated=17
- [PASS] **CTT unrelated incorrect-merge limitation included** — false_campaign=0; fleet_campaign=0; campaign_F1=0; incorrect_merge=1; membership_F1=1
- [PASS] **No OCSLab result file modified**
- [PASS] **No can-train-and-test result file modified**
- [PASS] **No heavy experiment rerun** — build_cross_dataset_comparison.py reads CSV/MD only

## Overall: PASS

> **Note:** OCSLab root absent from workspace. OCSLab numeric cells use `SOURCE_NOT_IN_WORKSPACE`. Sync `new_experiments/final_end_to_end_publication_run` and re-run `python scripts/build_cross_dataset_comparison.py` to populate OCSLab columns.
