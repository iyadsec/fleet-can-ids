# Result provenance — resolving draft vs baseline-ablation conflicts

## Three OCSLab result sets (do not mix without labelling)

| Result set | Branch path | Strong F1 (cs=5) | Unrelated merge | Status |
|------------|-------------|------------------|-----------------|--------|
| **A. Draft Tables III–IV (original split)** | pre-balanced end-to-end run | **0.867** | **0.400** | Superseded — draft not yet updated |
| **B. Balanced publication (authoritative)** | `final_end_to_end_publication_run_balanced/` | **0.733** | **0.400** | **Use for Section VII primary results** |
| **C. Framework ablation (M1/M2/M3 compare)** | `publication_ready/table_06` + `results/S2_*` | **0.406** (fcgnn) | **0.000** (fcgnn) | Descriptive ablation only; different eval correction |

### Why draft Table IV (0.867) ≠ baseline-ablation (0.406 or 0.733)

1. **Draft 0.867** comes from the **original train/validation split** before the balanced Chevrolet-inclusive split.
   See `audit/original_vs_balanced_split.md`: original strong cs=5 F1 = 0.867 → balanced = 0.733.

2. **Baseline-ablation 0.406** (previous bundle) pulled `table_06_method_ablation.csv` from the **framework ablation**
   pipeline with evaluation-correction / promotion rules applied on phase-2/3 scenario artifacts — not the balanced
   end-to-end publication run.

3. **Corrected baseline-ablation M3** now uses the **balanced publication run** (0.733 strong F1, 0.400 unrelated merge),
   aligning with draft Table III (merge) and the authoritative Section VII numbers (F1).

### Why draft Table III (0.400) ≠ old baseline-ablation (0.000)

- Draft **0.400** = `table_P6` unrelated multi-vehicle incidents from the **balanced publication** pipeline.
- Old baseline **0.000** = `S2_non_coordinated/summary_mean_std.csv` under **framework ablation** (different scenario
  construction, metric definition, and fcgnn slice at campaign_size=5, coord=0).

### Final publication version

**Use B (balanced publication)** for all primary OCSLab fleet results in Section VII.
Update draft Table IV from 0.867 → 0.733 (and cs=10 from 0.933 → 1.000) when revising the manuscript.

**Use C (framework ablation)** only for the explicit M1 vs M2 vs M3 ablation subsection — with M3 fleet metrics
sourced from B when reporting headline coordinated-campaign performance.

Cross-dataset (can-train-and-test) results remain in `OVERLEAF_CROSS_DATASET_ARTIFACTS/` and `experimental-2026-06-19/03_cross_dataset_ctt/`.
