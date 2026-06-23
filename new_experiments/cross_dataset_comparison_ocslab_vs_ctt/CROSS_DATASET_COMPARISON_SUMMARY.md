# Cross-Dataset Comparison Summary

**Output root:** `new_experiments/cross_dataset_comparison_ocslab_vs_ctt`  
**Validation:** PASS  
**Generated:** 2026-06-20 21:14 UTC

## 1. What OCSLab results were compared?

OCSLab publication root not present in workspace; dataset-level metadata from code; performance metrics marked `SOURCE_NOT_IN_WORKSPACE`.

Expected source: `new_experiments/final_end_to_end_publication_run/` — publication tables (dataset summary, local detection, descriptors, graph statistics, scenario results, campaign-size and edge sensitivity).

## 2. What can-train-and-test results were compared?

Full cross-dataset validation run at `new_experiments/can_train_and_test_cross_dataset_validation/full/`:
- `CTT_PUBLICATION_RESULT_DIGEST.md`, `CTT_PUBLICATION_KEY_NUMBERS.csv`
- Pooled tables `table_CTT1`–`table_CTT10`
- Per-set graph statistics and validation reports

## 3. Which metrics are directly comparable?

See `audit/metric_compatibility_audit.md` (Class A): descriptor size, bandwidth reduction, candidate transmission rate, cross-vehicle edge percentage.

## 4. Which metrics are descriptive only?

Class B metrics: local precision/recall/F1, ROC/PR-AUC, graph nodes/edges, all fleet scenario outcomes, campaign-size and edge-sensitivity trends, runtime/memory.

## 5. Where does can-train-and-test confirm the main framework?

- End-to-end pipeline on 4 vehicles, 2 manufacturers, 9 attack types
- ~70% descriptor bandwidth reduction
- Benign false_campaign = 0; isolated attacks without fleet escalation
- Strong/weak campaign F1 = 1.0
- Behavioural graph with ~0.27% cross-vehicle edges

## 6. Where does can-train-and-test expose limitations?

- Local detection: low pooled F1 on cross-vehicle subsets despite high ROC-AUC
- Unrelated incidents: **incorrect_merge_rate = 1.0** (behaviour-only graph over-merge)
- No real synchronized fleet campaigns in either dataset

## 7. Which tables should go into the main paper?

- **Table COMP1** — dataset roles and coverage
- **Table COMP5** — scenario campaign outcomes (include unrelated limitation)
- **Table COMP8** — overall cross-dataset summary

## 8. Which figures should go into the main paper?

- **Figure COMP1** — dataset coverage
- **Figure COMP5** — scenario outcome comparison

## 9. Which comparison items should go to supplementary material?

Tables COMP2–COMP4, COMP6–COMP7; Figures COMP2–COMP4, COMP6–COMP7; full source map (`results/comparison_source_map.csv`).

## 10. What exact wording should be used in the paper?

See `audit/recommended_comparison_paper_wording.md`.

---

*All comparisons labelled as descriptive cross-dataset comparison unless metric definitions are confirmed identical.*
