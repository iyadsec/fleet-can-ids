# HISTORICAL_COORDINATION_ARTIFACT_INVENTORY.md

Scope: **historical publication / freeze / archive artifacts only**  
(P6/P7/P8, balanced publication run outputs, freeze configs).  
**Excluded:** `experimental-2026-09-20/eta_revalidation_from_raw/` reconstructed validation outputs.

## Search result (high level)

| Artifact class | Found in P7/P8 archive? |
|----------------|-------------------------|
| Post-coordination descriptor vectors | **No** |
| Graph node feature matrices | **No** |
| Embeddings | **No** |
| Edge lists / pairwise cosines | **No** |
| Cluster assignments / centroids | **No** (only aggregate `largest_cluster_size`, noise %) |
| Per-run scenario membership CSVs | **No** (MISSING_MODULES: `results/scenario_evaluation/runs/` never committed) |
| Prototype vectors / ids | **No** |
| Numeric `coordination_strength` | **No** |
| Aggregate campaign / graph metrics | **Yes** |

---

## Inventory

| path | scenario | strong/weak | campaign size | seed | fields available | pre/post coordination | useful for inferring strength? |
|------|----------|-------------|---------------|------|------------------|----------------------|--------------------------------|
| `experimental-2026-06-23/01_primary_ocslab_balanced/results/campaign_metrics.csv` (= frozen reference) | benign / isolated / unrelated / strong / weak | both via `attack_strength` / `test_condition` | 2,5,10 (campaigns); 5 for controls | 11…101 | F1/membership/graph aggregates; `largest_cluster_size`; `cross_vehicle_edges`; `dbscan_noise_percentage`; **no** cohesion, cosine, features, strength | post-pipeline aggregates only | **Indirect only** — cannot test s=1 collapse |
| `…/results/strong_summary.csv` | strong | strong | 2,5,10 | aggregated | same metric family | aggregate | Indirect only |
| `…/results/weak_summary.csv` | weak | weak | 2,5,10 | aggregated | same | aggregate | Indirect only |
| `…/tables/table_P7_strong_campaign_results.csv` | strong (P7) | strong | 2,5,10 | mean seed 58.4 rows | same as strong summary | aggregate | Indirect only |
| `…/tables/table_P8_weak_campaign_results.csv` | weak (P8) | weak | 2,5,10 | mean seed rows | same | aggregate | Indirect only |
| `…/tables/table_P6_…csv` | non-campaign | n/a | — | — | safety / merge metrics | aggregate | Not relevant to s |
| `…/tables/table_P9_campaign_size_graph_and_cost.csv` | strong/weak | both | 2,5,10 | — | edges, latency, nodes | aggregate | Indirect weak |
| `…/tables/table_P5_descriptor_compactness_and_privacy.csv` | global | n/a | — | — | byte sizes / compression only | n/a | **No** feature values |
| `…/tables/verified_{strong,weak}_campaign_detection.csv` | strong/weak | both | 2,5,10 | — | detection/P/R/F1 consistency | aggregate | **No** (performance; do not use to fit s) |
| `recovered_publication_pipeline/frozen_results_reference/*` | copies of above | | | | identical campaign_metrics | | same |
| `…/configs/balanced_master_experiment.yaml` | config | — | campaign_sizes 2/5/10 | seeds listed | `behavioural_coordination_only: true`; **no** numeric strength | config | Confirms coordination **used**; not which \(s\) |
| `…/configs/final_shared_fleet_configuration.yaml` | freeze | — | — | — | graph/DBSCAN/gate; **no** `coordination_strength` | config | **Not** useful for \(s\) |
| `…/validation_scenarios/validation_manifest.csv` | V0–V4 (+ cs stubs) | strong/weak for V3/V4 | 0/1/5 (+ stub 2/10) | 131…197 | traces, hashes, event_id lists (often); **no** feature vectors; **no** strength | pre-fleet scenario metadata | Cannot test collapse |
| `…/validation_scenarios/parameter_search.csv` / `selection_report.md` | validation gate search | V3/V4 F1 reported | — | — | gate params + F1 | post-eval aggregates | **Do not** fit \(s\) from F1 |
| `…/scalers/fleet_benign_scaler.json` | scaler | — | — | — | means/stds for 9-D | pre-blend normalization later | Not strength |
| Per-run `results/scenario_evaluation/runs/*` | — | — | — | — | **Absent from git** | — | Would have been decisive if present |

### Explicit absences

- No saved `d'_i[C]` tables → **cannot** measure within-campaign variance / pairwise equality on transformed coordinates.
- `malicious_cross_vehicle_edge_purity` column exists but is **entirely NaN** in archived `campaign_metrics.csv` (0 non-null values).
- No logs/command lines/argparse dumps in the balanced archive recording `coordination_strength=…`.

---

## Direct configuration evidence classification

| Source | Mentions | S3 | S4 | Classification |
|--------|----------|----|----|----------------|
| `balanced_master_experiment.yaml` `behavioural_coordination_only: true` | coordination on | applies | applies | **DIRECT_PUBLICATION_EVIDENCE** that blending was enabled; **not** a numeric \(s\) |
| `final_shared_fleet_configuration.yaml` | no strength key | — | — | **NOT_RELEVANT** to numeric \(s\) |
| P7/P8 / campaign_metrics columns | no strength | — | — | **NOT_RELEVANT** to numeric \(s\) |
| `scenario_registry.py` ranges `(0.75, 1.0)` | range | yes | yes | **ALLOWED_RANGE_ONLY** (generic peer; not proven = missing `PUBLICATION_SCENARIOS`) |
| Peer `validation_scenarios.py` / baseline hardcode `1.0` | numeric | yes | yes | **PEER_IMPLEMENTATION** — **not** P7/P8 artifact evidence |
| Stand-in `0.35` | numeric | — | weak | **NOT_RELEVANT** (prospective stand-in) |
