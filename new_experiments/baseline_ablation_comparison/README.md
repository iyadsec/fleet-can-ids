# Baseline and ablation comparison

## Status

**Results assembled from existing experiments** on branch `origin/cursor/campaign-clustering`.
No experiments were rerun for this bundle.

## Methods

| Paper ID | Framework key | Description |
|----------|---------------|-------------|
| M1_Local_IF_Only | `local_ids` | Vehicle-level Isolation Forest only |
| M2_Descriptor_Clustering_Only | `descriptor_clustering` | Descriptor DBSCAN (no GNN) |
| M3_GraphSAGE_Fleet_Model | `fcgnn` | Proposed GraphSAGE fleet correlator |

## Scenarios

| Scenario | Role | Primary slice |
|----------|------|---------------|
| S2_non_coordinated | Independent multi-vehicle attacks | `campaign_size=5`, `coordination_strength=0.0` |
| S3_strong_campaign | Strong coordinated campaign | Publication `table_06` (campaign size 5) |
| S4_weak_campaign | Weak coordinated campaign | Publication `table_06` (campaign size 5) |
| S0_benign_control | Benign / false campaign (supplementary) | `campaign_size=0` |

Seeds (paper): 11, 23, 37, 41, 53, 67, 71, 83, 97, 101.

## Source files

- `new_experiments/publication_ready/tables/table_06_method_ablation.csv` (table_06)
- `new_experiments/results/S2_non_coordinated/summary_mean_std.csv` (S2_summary)
- `new_experiments/results/S3_strong_campaign/summary_mean_std.csv` (S3_summary)
- `new_experiments/results/S4_weak_campaign/summary_mean_std.csv` (S4_summary)
- `new_experiments/results/S0_benign_control/summary_mean_std.csv` (S0_summary)
- `new_experiments/results/S3_strong_campaign/run_level_metrics.csv` (S3_runs)
- `new_experiments/results/S4_weak_campaign/run_level_metrics.csv` (S4_runs)
- `new_experiments/results/S2_non_coordinated/run_level_metrics.csv` (S2_runs)

## Metric definitions (where calculated)

- **Local vehicle recall**: `Vehicle recall` in `table_06_method_ablation.csv` (S3); local IF detection only.
- **Strong / weak campaign F1**: `Campaign F1` in `table_06_method_ablation.csv` for S3/S4.
  M1 has no fleet layer → campaign F1 reported as N/A in the paper table (0 by definition).
- **Independent incorrect merge**: `incorrect_campaign_merging_mean` in
  `S2_non_coordinated/summary_mean_std.csv` (unrelated attacks must not merge).
- **False campaign (S0)**: `false_campaign_alert_rate_mean` in `S0_benign_control/summary_mean_std.csv`.
- **Runtime**: mean `runtime_total_sec_mean` from S3/S4 summaries (`campaign_size=5`, `coord=1.0`).

Ground-truth campaign labels come from controlled scenario assignment only; attack types are not model inputs.

## Outputs

- `results/baseline_ablation_metrics.csv` — full numeric metrics
- `tables/table_baseline_ablation.csv` — paper-ready CSV
- `tables/table_baseline_ablation.tex` — IEEE-friendly LaTeX
- `figures/figure_baseline_ablation_campaign_f1.pdf` — campaign F1 bar chart

## Reproduce

```bash
python scripts/build_baseline_ablation_comparison.py
```
