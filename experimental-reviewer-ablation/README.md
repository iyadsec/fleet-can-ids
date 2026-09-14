# Controlled Reviewer-A Ablation (NEW)

This directory contains a **new, controlled component ablation** answering Reviewer A
(Sec 4.5.2). Absolute metrics are **not** claimed identical to frozen historical
publication tables.

## Runnable stack selected

| Stage | Implementation |
|---|---|
| Runner | `experimental-reviewer-ablation/run_ablation.py` |
| Scenarios | Synthetic controlled descriptors (`scenario_builder.py`) |
| Descriptor | 9-D behavioural features (anomaly score + 8 traffic stats) |
| Graph | `src.graph.scenario_graph.build_scenario_graph_from_features` (cosine τ=0.95, same-k=2, cross-k=5) |
| GraphSAGE | `src.models.gnn_models.train_graphsage_fleet_correlation` (9→64→32, structure loss, λ=0.25) |
| GCN | Local `GCNFleetCorrelator` mirror with the same objective |
| Clustering | `src.evaluation.campaign_clustering.run_dbscan` (Scaler→PCA(8)→DBSCAN eps=0.5, min_samples=2) |
| Campaign gate | `summarize_clusters` (min vehicles=2, cohesion=0.85) + fragment merge 0.85 |
| Metrics | Jaccard≥0.5 campaign matching (`metrics.py`) |

## Variants

- **M1 Local IF** — vehicle-level alerts only; campaign metrics = N/A
- **M2 Descriptor clustering** — descriptors → Scaler/PCA/DBSCAN → campaign gate
- **M3 GCN** — same graph → 2-layer GCN → Scaler/PCA/DBSCAN → gate
- **M4 GraphSAGE / FLEET-GUARD** — same graph → GraphSAGE → Scaler/PCA/DBSCAN → gate

## Control protocol

For each seed, one scenario is frozen and shared by all methods.
M2/M3/M4 receive identical descriptor IDs and GT campaigns.
M3/M4 receive identical graphs (asserted in `verification_report.txt`).

## Reproduce

```bash
# smoke (seed 11)
python3 experimental-reviewer-ablation/run_ablation.py --smoke

# full (10 seeds)
python3 experimental-reviewer-ablation/run_ablation.py
```

## Outputs

- `raw_results.csv` — per seed / scenario / method
- `summary_results.csv` — mean ± std
- `ablation_table.csv` — publication-ready table
- `verification_report.txt` — shared scenario / graph assertions
- `provenance.json` — environment fingerprint
- `paper_ablation_subsection.tex` — paper-ready text
- `reviewer_response.md` — draft reviewer reply
