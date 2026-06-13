# Final scenario experiment summary

**Package:** `new_experiments/final_publication_scenarios/`  
**Generated from authoritative sources only** (see `audit/source_selection_report.md`).  
**Vehicle-model-diversity and CICIoV2024 excluded.**

## 1. Did S0 avoid false campaign alerts?

Yes under C3 GraphSAGE: mean false campaign alert rate = **0.0** across ten seeds (`results/scenarios/scenario_safety_metrics.csv`). Local false-positive rate on benign events remains non-zero (mean FPR ≈ 0.41 on local IDS); fleet layer did not declare campaigns.

## 2. Did S1 remain an isolated incident?

Yes: local attacked-vehicle recall = **1.0**; isolated-incident decision rate = **1.0**; incorrect campaign declaration rate = **0.0**.

## 3. Did S2 avoid incorrect campaign merging?

Partially: attacked-vehicle detection = **1.0**; incorrect merging rate = **0.33**; false single-campaign declaration rate ≈ **0.48**. Separation is imperfect but non-trivial merging resistance is observed.

## 4. Did S3 detect strong coordinated campaigns?

Yes: campaign detection rate ≈ **0.81**; campaign F1 ≈ **0.69**; campaign recall ≈ **0.81** (C3, hierarchical alignment).

## 5. Did S4 correlate weak distributed evidence?

Yes with lower margin: campaign detection rate ≈ **0.72**; campaign F1 ≈ **0.67**; weak attacked vehicles correlated (see `weak_campaign_support.csv` and `figure_F1_weak_campaign_support`).

## 6. How did campaign size affect detection?

Corrected 200-node fcgnn runs (`campaign_size_corrected`): strong F1 peaks at n=2 (0.60) then declines at n=10 (0.30); weak F1 increases with size (n=2: 0.60 → n=10: 0.90). Detection rate follows similar trends. See `table_T5` / `table_T6` and `figure_F2` / `figure_F3`.

## 7. How did campaign size affect edge count and cost?

Unique edges and graph-build time increase with fleet/campaign scale (see `table_T7_campaign_size_cost`). Full graph statistics in `results/campaign_size/graph_statistics.csv`.

## 8. How did measured edge count affect campaign F1?

Edge sweep (400 runs, fixed 200-node records): unique edges ranged **370–1311**. S3 peak mean F1 near **~437 edges**; S4 peak near **~370 edges**. See `figure_F4_campaign_F1_vs_unique_edges_S3/S4`.

## 9. Under-connected graphs?

Lower thresholds / smaller k produce fewer cross-vehicle edges, higher fragmentation risk, and reduced recall in some bins (see low-edge tail in edge summary).

## 10. Over-connected graphs?

Higher thresholds with large k approach denser graphs; membership precision and false-alert context should be read jointly (`figure_F5`).

## 11. Best measured edge trade-off?

Empirically **370–500 unique undirected edges** for S4/S3 respectively under the fixed 200-node campaign-size-5 scenarios, balancing F1 against false campaign rate. Full grid in `results/edge_sensitivity/run_level_metrics.csv`.

## 12. Tables and figures ready?

All T1–T10 and F1–F6 generated under `tables/` and `figures/`. See `validation/publication_artifact_completeness.md`.

## 13. Limitations

- Legacy `false_campaign_alert_rate` in raw Phase 2/3 metrics can equal 1.0 when any cluster is detected; hierarchical C3 metrics used for publication safety tables.
- S0–S2 Phase 2 scenarios use variable node counts; campaign-size and edge sweeps use fixed 200-node corrected compositions.
- Statistical tests may be non-significant with n=10 seeds and high variance (`table_T10`).
- No values hard-coded; all aggregates computed from source CSVs.
