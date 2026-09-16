# Final report — controlled Reviewer-A ablation

## A. Runnable pipeline selected

- **Runner:** `experimental-reviewer-ablation/run_ablation.py`
- **Dataset artifacts:** synthetic controlled 9-D suspicious descriptors (no OCSLab dump required)
- **Scenario generator:** `scenario_builder.py` (`strong_campaign`, `weak_campaign`, `unrelated_incidents`)
- **Descriptor representation:** 9-D behavioural features listed in `config.yaml`
- **Graph builder:** `src.graph.scenario_graph.build_scenario_graph_from_features` (cosine τ=0.95, same-k=2, cross-k=5)
- **GraphSAGE:** `src.models.gnn_models.train_graphsage_fleet_correlation` (9→64→32, structure + λ=0.25)
- **Clustering:** `src.evaluation.campaign_clustering.run_dbscan` (Scaler→PCA 8 → DBSCAN eps=0.5, min_samples=2)
- **Campaign evaluation:** Jaccard≥0.5 matching in `metrics.py` after `summarize_clusters` gate (min vehicles=2, cohesion=0.85, fragment merge=0.85)

## B. Exact ablation configuration

Frozen in `config.yaml`: 20 vehicles × 10 windows = 200 nodes; campaign size 5;
seeds `[11,23,37,41,53,67,71,83,97,101]`; graph/GNN/clustering/gate as above.

## C. Definition of M1–M4

- **M1:** local IF alerts only; no graph/GNN/DBSCAN campaigns → campaign metrics N/A
- **M2:** descriptors → Scaler/PCA/DBSCAN → campaign gate
- **M3:** shared similarity graph → 2-layer GCN (same dims/objective) → Scaler/PCA/DBSCAN → gate
- **M4:** shared graph → GraphSAGE → Scaler/PCA/DBSCAN → gate

## D. Verification that scenarios and node sets were shared

`verification_report.txt`: all 10 seeds × 3 scenarios PASS with `shared_ids=OK` and `graph_m3==graph_m4`.

## E. Raw numerical results

See `raw_results.csv` (120 rows = 10 seeds × 3 scenarios × 4 methods).

## F. Mean ± std table

| Method | Strong Campaign F1 | Weak Campaign F1 | Strong Membership F1 | Weak Membership F1 | Independent Merge Rate |
|---|---|---|---|---|---|
| M1 Local IF | N/A | N/A | N/A | N/A | N/A |
| M2 Descriptor Clustering | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.981 ± 0.015 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| M3 GCN | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| M4 GraphSAGE / FLEET-GUARD | 0.600 ± 0.516 | 0.500 ± 0.527 | 0.543 ± 0.474 | 0.425 ± 0.454 | 0.000 ± 0.000 |

(Also in `ablation_table.csv`. M1 strong local F1 = 1.000 ± 0.000.)

## G. Exact metric definitions discovered in code

From `metrics.py` / gate in `methods.py` + `summarize_clusters`:

- **Match:** greedy 1–1 Jaccard ≥ 0.5 between predicted and GT campaign node sets
- **Campaign Precision / Recall / F1:** matched/pred, matched/GT, harmonic mean
- **Membership F1:** micro F1 over node membership of matched pairs (unmatched GT→FN, unmatched pred→FP)
- **Fragmentation:** mean over GT campaigns of (# overlapping predicted clusters)
- **Incorrect merge rate:** fraction of predicted campaigns spanning ≥2 GT campaign IDs
- **False campaign rate:** fraction of predicted campaigns with no GT match

## H. Interpretation

1. **Descriptor clustering already useful?** Yes for strong campaigns (F1=1.0); not for weak (F1=0).
2. **Graph representation learning improve over clustering?** Mixed: GraphSAGE helps weak campaigns (0.5 vs 0) but is worse/noisier than M2 on strong campaigns (0.6±0.52 vs 1.0).
3. **GraphSAGE over GCN?** Yes — M4 recovers campaigns; M3 does not (F1=0) and merges unrelated attacks.
4. **Incorrect merging of unrelated attacks?** Increased only for M3 GCN (rate 1.0); M2/M4 stay at 0.
5. **Stronger for strong or weak?** Relative GraphSAGE gain vs M2 is clearer for **weak** campaigns; on strong campaigns M2 already saturates.

We do **not** claim unconditional GraphSAGE superiority over descriptor clustering.

## I. Paper-ready Ablation Study subsection

See `paper_ablation_subsection.tex`.

## J. Reviewer-response paragraph

See `reviewer_response.md`.

## K. Discrepancies vs historical publication configuration

1. Synthetic controlled descriptors (not frozen OCSLab publication dumps).
2. Historical publication runner packages were never committed; this uses the current runnable stack.
3. Absolute metrics are not numerically comparable to `01_primary_ocslab_balanced` tables.
4. GCN baseline is new for this ablation.
5. Metrics use documented Jaccard matching in this directory (publication metric modules absent from git).
