# Source selection report

## Authoritative sources

### S0–S4 scenario evaluation
- Primary run metrics: `new_experiments/final_validated_runs/results/S{0-4}_*/run_level_metrics.csv`
- Hierarchical aligned metrics: `new_experiments/final_validated_runs/hierarchical_alignment/results/`

### Campaign-size sensitivity (corrected)
- `new_experiments/final_validated_runs/results/campaign_size_corrected/run_level_metrics.csv`
- Per-run scenario records: `.../campaign_size_corrected/runs/*/selected_source_records.csv` (200 nodes each)

### Edge connectivity sensitivity
- New authoritative runs under `new_experiments/final_publication_scenarios/results/edge_sensitivity/`
- Fixed scenario records reused from S3/S4 fcgnn baseline (campaign size 5, coordination 1.0)

## Excluded roots

- **new_experiments/final_validated_runs/results/campaign_size** (preliminary): variable node counts; superseded
- **new_experiments/final_validated_runs/framework_ablation** (final): supplementary; not primary scenario tables
- **new_experiments/final_validated_runs/evaluation_correction** (corrected_metrics): campaign-size only; layered correction
- **new_experiments/final_validated_runs/model_diversity** (preliminary): split leakage; excluded
- **new_experiments/final_validated_runs/model_diversity_corrected** (diagnostic): superseded by model_diversity_final
- **new_experiments/final_validated_runs/model_diversity_final** (final): vehicle-model diversity; out of scope
- **new_experiments/final_validated_runs/model_diversity_final_tuned** (diagnostic): gate tuning provisional; out of scope
- **new_experiments/final_validated_runs/quick_test** (diagnostic): smoke test seed 11 only

## Policy

- No aggregation across preliminary and corrected campaign-size versions.
- Vehicle-model-diversity experiments excluded from this package.
- Local (C1) and fleet (C2/C3) metrics remain separated.