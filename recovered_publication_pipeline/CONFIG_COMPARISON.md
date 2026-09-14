# Configuration comparison — recovered freeze vs required publication checklist

Sources under `recovered_publication_pipeline/`:

- `new_experiments/final_end_to_end_publication_run_balanced/configs/balanced_master_experiment.yaml`
- `new_experiments/final_end_to_end_publication_run_balanced/configs/final_shared_fleet_configuration.yaml`
- `frozen_results_reference/campaign_metrics.csv` (+ P6/P7/P8 tables)

Values are reported as recovered; **not adjusted**.

| Parameter | Required (audit checklist) | Recovered freeze | Match |
|-----------|----------------------------|------------------|-------|
| Fleet vehicles | 20 | 20 (`scenario.fleet_size`) | YES |
| Windows / vehicle | 10 | 10 (`scenario.source_windows_per_vehicle`) | YES |
| Total nodes | 200 | 200 (`scenario.total_source_windows`) | YES |
| Seeds | 11,23,37,41,53,67,71,83,97,101 | same in `local_ids.random_seeds` and `statistics.seeds` | YES |
| Campaign sizes | 2,5,10 | `[2,5,10]` | YES |
| Scenarios | benign / isolated / unrelated / strong / weak | Present in frozen `campaign_metrics.csv` (`scenario_key` counts 10+10+10+30+30) | YES (artifact); scenario class defs live in **MISSING** `scenario_registry` |
| Cosine threshold | 0.95 | 0.95 (`similarity_threshold`) | YES |
| Same-vehicle cap | 2 | 2 (`max_same_vehicle_neighbors`) | YES |
| Cross-vehicle cap | 5 | 5 (`max_cross_vehicle_neighbors`) | YES |
| GNN input dim | 9 | Not in YAML; `prepare_gnn_fleet_node_matrix` in recovered eval code | CODE-LEVEL (recovered) |
| GraphSAGE 9→64→32 | hidden 64; out 32 | YAML: `graphsage.hidden_dim=64`; layer shape in recovered `gnn_models.py` | YES for hidden; out dim from code |
| Mean aggregation | mean | PyG SAGEConv default in recovered `gnn_models.py` | YES (code) |
| ReLU after first layer | yes | Present in recovered GraphSAGE class | YES (code) |
| λ | 0.25 | Not in master YAML | CODE-LEVEL / largely inside **MISSING** refinement pipeline |
| Adam lr | 0.01 | 0.01 | YES |
| Weight decay | 5e-4 | Not in master YAML | CODE-LEVEL / MISSING refinement |
| Epochs | 30 | 30 | YES |
| StandardScaler + PCA 8 | yes | PCA=8 default in recovered eval experiment; fleet benign scaler JSON recovered | PARTIAL (freeze application via missing SharedFleetConfiguration) |
| DBSCAN eps / min_samples | 0.5 / 2 | **Freeze YAML: 0.5 / 2**; recovered eval **class defaults still 1.2 / 10** | FREEZE YES; code defaults differ — freeze must be applied by missing config consumer |
| Min vehicles | 2 | 2 | YES |
| Min cross support | 1 | 1 | YES |
| Min cohesion | 0.5 | 0.5 | YES |
| Fragment centroid | 0.85 | 0.85 | YES |

## Frozen aggregate check (reference only; not a re-run)

From recovered `campaign_metrics.csv` (method=`fcgnn`):

| Scenario | campaign_size | mean campaign_f1 |
|----------|---------------|------------------|
| strong_campaign | 2 / 5 / 10 | 0.533 / 0.733 / 1.000 |
| weak_campaign | 2 / 5 / 10 | 0.067 / 0.500 / 0.717 |
| unrelated_incidents | (placeholder 5) | incorrect_merging 0.400 (P6) |
| benign / isolated | — | false campaign rate 0.000 (P6) |

These match `BALANCED_PUBLICATION_SUMMARY.md` on `campaign-clustering`.

## Discrepancies (do not “fix”)

1. Master YAML does not encode λ, weight decay, embedding output dim, PCA dim, or scenario enum — those lived in missing packages / model code.
2. Recovered `FinalGnnFleetDecisionConfig` still defaults DBSCAN to **eps=1.2, min_samples=10**; publication numbers require the freeze file to be applied by `SharedFleetConfiguration` (**MISSING**).
3. Frozen metrics CSV reuses columns `method=fcgnn`, `framework_config=C3`; that is labeling/schema reuse, not proof the framework-ablation runner produced these rows.
