# Non-collaborative vs FLEET-GUARD validation

## Provenance
- FLEET-GUARD metrics: `/workspace/experimental-2026-06-23/01_primary_ocslab_balanced/results/campaign_metrics.csv` (read-only archive; not recomputed)
- P6 safety table: `/workspace/experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P6_benign_isolated_unrelated_results.csv`
- P7 strong campaign table: `/workspace/experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P7_strong_campaign_results.csv`
- P8 weak campaign table: `/workspace/experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P8_weak_campaign_results.csv`
- Baseline inputs: descriptors: not available (window reconstruction skipped); manifest: git show origin/cursor/campaign-clustering:new_experiments/final_end_to_end_publication_run_balanced/manifests/balanced_window_manifest.csv

## Validation checks
- Same scenario seeds (10 seeds): PASS
  - campaign_metrics seeds: [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
  - baseline seeds: [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
- FLEET-GUARD metrics sourced from validated archive only: PASS
- Vehicle-Level IDS baseline did not use graph, GraphSAGE, DBSCAN, or campaign clusters: PASS
- Baseline campaign metrics are N/A (campaign reasoning unsupported): PASS
- Baseline local metrics computed from per-window local_alert counts (strong_candidates), not descriptor promotion aggregates (benign_incorrectly_promoted): PASS

## Headline FLEET-GUARD campaign F1 (campaign_size=5)
- Strong (table P7): 0.733
- Weak (table P8): 0.500
- Strong (per-seed mean from campaign_metrics): 0.733
- Weak (per-seed mean from campaign_metrics): 0.500
- Benign false campaign rate: 0.000

## Baseline local IDS (scenario means, theta_strong)
- S0: P=nan, R=nan, F1=nan, FPR=0.000
- S1: P=1.000, R=1.000, F1=1.000, FPR=0.000
- S2: P=1.000, R=1.000, F1=1.000, FPR=0.000
- S3: P=0.552, R=1.000, F1=0.711, FPR=0.117
- S4: P=1.000, R=0.832, F1=0.907, FPR=0.000

## Publication readiness
READY for scenario comparison tables when FLEET-GUARD archive tables validate and baseline local metrics use per-window theta_strong alerts (not descriptor candidate promotion).
