# Curated Cross-Dataset Comparison Summary

**Output root:** `new_experiments/cross_dataset_comparison_ocslab_vs_ctt_curated`  
**Validation:** PASS  
**Generated:** 2026-06-20 22:12 UTC

## 1. OCSLab sources used

- `paper/results/` (curated IEEE export from `origin/cursor/campaign-clustering`): vehicle-level Table 1, descriptor Table 2, full fleet graph statistics
- `new_experiments/final_publication_scenarios/` (final scenario package): S0–S4 safety metrics, edge sensitivity 370–1311 on 200-node graphs

**Not used:** `final_end_to_end_publication_run_balanced/`

## 2. CTT sources used

- `new_experiments/can_train_and_test_cross_dataset_validation/full/` pooled tables, key numbers CSV, digest, validation reports

## 3. Tables generated

CUR_COMP1–CUR_COMP5 (CSV, Markdown, LaTeX)

## 4. Figures generated

CUR_COMP1 (coverage), CUR_COMP2 (bandwidth reduction), CUR_COMP3 (scenario outcomes) — PNG and PDF

## 5. Key positive findings

- Framework applies to CTT: 4 vehicles, 2 manufacturers, 9 attack types
- Descriptor bandwidth reduction on both datasets (~92% OCSLab paper; ~70% CTT)
- Benign/isolated scenario safety on CTT; strong/weak campaign F1 = 1.0 pooled
- OCSLab primary headline metrics preserved via curated paper export

## 6. Key limitations

- **CTT unrelated incorrect_merge_rate = 1.0**
- Local detection not benchmark-equivalent (CTT low pooled F1 despite high ROC-AUC)
- Graph scales and edge sweeps differ (200-node OCSLab scenario package vs ~100k-node CTT graphs)

## 7. Main paper table

**Table CUR_COMP3** (fleet scenario comparison) — include unrelated limitation explicitly

## 8. Main paper figure

**Figure CUR_COMP3** (scenario outcomes) or **Figure CUR_COMP1** (coverage)

## 9. Supplementary material

CUR_COMP2, CUR_COMP4, CUR_COMP5; Figure CUR_COMP2; full `curated_source_map.csv`

## 10. Ready for paper?

**Yes** — curated descriptive cross-dataset comparison is ready using confirmed export sources. Label all comparisons as descriptive, not strict benchmarks.
