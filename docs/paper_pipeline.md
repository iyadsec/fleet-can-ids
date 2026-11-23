# Paper-to-code mapping (FLEET-GUARD)

This table maps paper experiments to runnable scripts and canonical output locations. **Primary OCSLab publication numbers** are archived in `experimental-2026-06-23/01_primary_ocslab_balanced/` (balanced split run). Re-running the full pipeline requires local datasets (see [datasets.md](datasets.md)).

The archived P4–P12 tables were produced by `scripts/run_balanced_publication_run.py` (commit `61a8203`, branch `cursor/campaign-clustering`), not by the numbered reference pipeline below. Executed parameters, unpublished modules and corrected claims: [RESULTS_PROVENANCE_AND_CORRECTIONS.md](RESULTS_PROVENANCE_AND_CORRECTIONS.md).

## Pipeline stages (implementation order)

| Step | Paper concept | Script / module | Default output |
|------|---------------|-----------------|----------------|
| 1 | Load raw CAN traces | `experiments/01_load_dataset.py` | `data/processed/clean_can_data.csv` |
| 2 | Parse and clean frames | `src/data/dataset_loader.py` | (merged into clean CSV) |
| 3 | Segment into 100-frame windows | `experiments/02_generate_windows.py` | `data/processed/window_metadata.csv` |
| 4 | Behavioural window features | `experiments/03_extract_features.py` | `data/processed/window_features.csv` |
| 5 | Benign-only Isolation Forest (unsupervised) | `experiments/04_train_vehicle_ids.py` | `outputs/metrics/vehicle_level_self_supervised_results.csv` (legacy path) |
| 6 | Anomaly scores | `src/models/vehicle_ids.py` | `data/processed/vehicle_anomaly_predictions.csv` |
| 7 | Strong local alerts | `experiments/04_train_vehicle_ids.py` (strong threshold) | `local_alert` column |
| 8 | Weak anomaly descriptors | `experiments/05_generate_descriptors.py` | `data/processed/anomaly_descriptors.csv` |
| 9 | Fleet behavioural graph (constrained k-NN; τ=0.95, same=2, cross=5) | `experiments/06_build_graph.py` → `src/graph/scenario_graph.py` | `data/processed/fleet_graph.pt` |
| 10 | GraphSAGE fleet correlation | `experiments/07_train_gnn.py` | `data/processed/node_embeddings.csv` |
| 11 | Graph embeddings | `src/models/gnn_models.py` | `outputs/metrics/gnn_training_metrics.csv` |
| 12 | DBSCAN clustering | `experiments/08_cluster_campaigns.py` | `data/processed/fleet_cluster_results.csv` |
| 13 | Campaign decision logic | `experiments/09_final_decision.py` | `outputs/metrics/final_detection_outcomes.csv` |
| 14 | Tables, figures, manifests | `experiments/09_generate_research_outputs.py` | `outputs/metrics/`, `outputs/figures/` |

**End-to-end OCSLab runner:** `python experiments/run_full_pipeline.py --config configs/default.yaml`

**Cross-dataset (can-train-and-test):** `python scripts/run_can_train_and_test_cross_dataset.py --stage full`

## Paper results mapping

| Paper result | Script | Output file / folder |
|--------------|--------|-------------------|
| Vehicle-level anomaly detection (Table P4, balanced fleet run) | `experiments/04_train_vehicle_ids.py` or balanced publication run | `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` |
| Vehicle-level anomaly detection (Table I, FPR≤5%) | `scripts/run_vehicle_level_fpr_controlled.py` | `results/vehicle_level_threshold_comparison.csv`, `tables/table_vehicle_level_ids.tex` |
| Per-vehicle Table I (Chevrolet, Hyundai, Kia, pooled) | `scripts/generate_per_vehicle_fpr_table.py` | `results/per_vehicle_validation_report.md` |
| Descriptor compactness (Table P5; 53.69% superseded, see provenance document) | `experiments/05_generate_descriptors.py` + privacy evidence | `.../table_P5_descriptor_compactness_and_privacy.csv` |
| Non-campaign scenarios (Table P6) | Fleet scenario evaluation | `.../table_P6_benign_isolated_unrelated_results.csv` |
| Strong campaign detection (Table P7) | Campaign scenarios + DBSCAN | `.../table_P7_strong_campaign_results.csv` |
| Weak campaign detection (Table P8) | Campaign scenarios + DBSCAN | `.../table_P8_weak_campaign_results.csv` |
| Campaign-size sensitivity (Table P9) | Campaign size sweep | `.../table_P9_campaign_size_graph_and_cost.csv` |
| Edge-connectivity sensitivity (Table P10) | Graph edge sweep | `.../table_P10_edge_connectivity_performance.csv` |
| Statistical analysis (Table P12) | Holm-corrected tests | `.../table_P12_primary_statistical_tests.csv` |
| Strong vs weak F1 figure (Fig. P4) | `scripts/build_figure_P4_strong_vs_weak_campaign_f1.py` | `.../figures/figure_P4_strong_vs_weak_campaign_F1.pdf` |
| Baseline/ablation (M1/M2/M3) | `scripts/build_baseline_ablation_comparison.py` | `experimental-2026-06-23/02_baseline_ablation/` |
| Cross-dataset local IDS comparison | `scripts/run_can_train_and_test_cross_dataset.py` | `OVERLEAF_CROSS_DATASET_ARTIFACTS/tables_csv/LOCAL_COMP*.csv` |
| Cross-dataset fleet correlation (ground-truth-assisted scoring) | CTT full stage + artifact builder | `OVERLEAF_CROSS_DATASET_ARTIFACTS/` |
| Fleet correlation outcomes figure (consistency rule is an oracle) | `scripts/build_figure_CORR_EFF3_consistency_rule_ablation.py` | `OVERLEAF_CROSS_DATASET_ARTIFACTS/figures_pdf/figure_CORR_EFF3_*.pdf` |
| June 2026 run reproduction (P4–P12) | `reproducibility/june_2026/reproduce.py`, `compare_with_june.py` | `reproducibility/june_2026/reproduction_check/` |

## Regenerate publication bundles

```bash
python scripts/verify_balanced_campaign_tables.py
python scripts/build_baseline_ablation_comparison.py
python scripts/consolidate_experimental_results.py
python scripts/build_overleaf_cross_dataset_artifacts.py
python scripts/regenerate_paper_artifacts.py
python scripts/validate_repository.py
```

## Notes for reviewers

- Coordinated campaigns are evaluated using **controlled fleet scenarios** because public CAN datasets do not provide naturally recorded synchronized fleet-wide attacks. In S3/S4 a shared malfunction-derived prototype is applied to attacked participants' descriptors before graph construction; detection then runs unchanged.
- The fleet graph and GraphSAGE use the 9-D scaled descriptor view; campaign cohesion uses a behaviour view of up to 8 features, from which features with a between-/within-vehicle variance ratio > 5 in the scenario are dropped (6–8 features in the June run).
- Attack labels are used **only for evaluation** and scenario assignment — not as model inputs during training or fleet inference.
- The cross-dataset validation dataset is **can-train-and-test** (DTU), not CAN-MIRGU.
- Descriptor abstraction is **privacy-aware** (reduced raw CAN exposure) but not formal privacy preservation.
