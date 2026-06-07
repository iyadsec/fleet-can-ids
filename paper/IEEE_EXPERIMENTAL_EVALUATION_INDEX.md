# IEEE Experimental Evaluation — Export Index

## Architecture (final)
Vehicle IDS → Anomaly Descriptors → Behaviour Graph → GraphSAGE (structure-only) → DBSCAN →
`isolated_attack` / `coordinated_attack`

## Tables
| Table | File | Hypothesis |
|-------|------|------------|
| Table 1 | `paper/tables/table_01_vehicle_level_ids.tex` | H1 — Vehicle-Level IDS |
| Table 2 | `paper/tables/table_02_descriptor_compactness_security.tex` | H2 — Descriptor Security |
| Table 3 | `paper/tables/table_03_cross_vehicle_generalisation.tex` | H3 — Cross-Vehicle Generalisation |
| Table 4 | `paper/tables/table_04_local_vs_gnn_fleet_ids.tex` | H4 — Fleet GNN IDS |
| Table 5 | `paper/tables/table_05_coordinated_campaign_detection.tex` | H4 — Campaign Detection |

## Figures
| Figure | File | Hypothesis |
|--------|------|------------|
| Figure 2 | `paper/figures/figure_02_vehicle_level_roc.pdf` | H1 |
| Figure 3 | `paper/figures/figure_03_anomaly_score_distribution.pdf` | H1 |
| Figure 4 | `paper/figures/figure_04_bandwidth_scaling.pdf` | H2 |
| Figure 5 | `paper/figures/figure_05_payload_reconstruction_risk.pdf` | H2 |
| Figure 6 | `paper/figures/figure_06_cross_vehicle_descriptor_embedding.pdf` | H3 |
| Figure 7 | `paper/figures/figure_07_gnn_fleet_campaign_graph.pdf` | H4 |
| Figure 8 | `paper/figures/figure_08_final_attack_decision_distribution.pdf` | H4 |
| Figure 9 | `paper/figures/figure_09_campaign_detection_by_attack_type.pdf` | H4 |

Figure 7 node colour = `isolated_attack` vs `coordinated_attack` (runtime decision).
Attack types in Figures 6 and 9 are evaluation/visualisation only.

## Supporting CSVs
All under `paper/results/`.

## Interpretations
`paper/results/ieee_experimental_evaluation_interpretations.md`
