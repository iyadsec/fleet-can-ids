# PUBLICATION_CONFIGURATION_AUDIT.md

**Purpose:** Distinguish (a) parameters actually used in the primary FLEET-GUARD OCSLab publication experiments, (b) general/current pipeline defaults, and (c) recoverable selection provenance — without inventing validation claims.

**Source-of-truth rule:** Archived publication experiment artifacts override `configs/default.yaml` when they conflict.

**Primary publication archive:** `experimental-2026-06-23/01_primary_ocslab_balanced/`  
**Authoritative freeze YAML (recovered pipeline tree):**  
`recovered_publication_pipeline/new_experiments/final_end_to_end_publication_run_balanced/configs/final_shared_fleet_configuration.yaml`  
(also summarized in that tree’s `validation_scenarios/selection_report.md` and `parameter_search.csv`)  
**Master experiment YAML:**  
`…/configs/balanced_master_experiment.yaml`  
**Master config hash:** `72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e`

**Companion file:** `configs/publication_fleet_guard.yaml`

---

## 1. Publication call / configuration chain (Task 1)

Documented publication runner path (recovered tree):

```text
recovered_publication_pipeline/scripts/run_balanced_publication_run.py
  → src.experiments.final_end_to_end_publication_run_balanced.runner
      → build_mixed_validation_suite(...)          [historically missing; see MISSING_MODULES.md]
      → joint_parameter_search(val_scenarios, …)   [historically missing module]
      → writes final_shared_fleet_configuration.yaml
      → _run_single_test / run_refinement_fcgnn    [historically missing]
```

On `main`, the paper-to-code map (`docs/paper_pipeline.md`) points general pipeline scripts at `configs/default.yaml`, while **archived Section VII numbers** live under `experimental-2026-06-23/01_primary_ocslab_balanced/`.

### Parameter evidence table

| Parameter | Publication value | Evidence location | How obtained |
|-----------|-------------------|-------------------|--------------|
| τ (`similarity_threshold`) | **0.95** | `final_shared_fleet_configuration.yaml`; `selection_report.md`; `src/graph/scenario_graph.py` defaults | Explicitly configured in freeze |
| k_same (`max_same_vehicle_neighbors`) | **2** | Same freeze / scenario_graph defaults | Explicitly configured in freeze |
| k_cross (`max_cross_vehicle_neighbors`) | **5** | Same freeze / scenario_graph defaults | Explicitly configured in freeze |
| GraphSAGE input dim | **9** | `GNN_FEATURE_COLUMNS` in recovered `final_gnn_fleet_decision_experiment.py`; audits list the 9 columns | Explicit feature schema in publication GNN prep |
| hidden dim | **64** | `balanced_master_experiment.yaml` `graphsage.hidden_dim`; `src/models/gnn_models.py` defaults; `configs/default.yaml` `gnn.hidden_channels` | Master YAML + code default |
| embedding dim | **32** | `gnn_models.py` / `default.yaml` `embedding_dim` (not listed in master YAML graphsage block) | Code / general config; consistent with paper |
| # SAGEConv layers | **2** | `GraphSAGEFleetCorrelator` / `GraphSAGEEncoder` in `src/models/gnn_models.py` | Hardcoded architecture |
| aggregation | **mean** | `torch_geometric.nn.SAGEConv` default (`aggr="mean"`) used without override | Inherited library default |
| activation | **ReLU after first layer** | `F.relu(self.conv1(...))` then second `SAGEConv` in `gnn_models.py` | Hardcoded |
| epochs | **30** | Master YAML `graphsage.epochs`; `gnn_models.train_*` defaults | Explicit + code default |
| learning rate | **0.01** | Master YAML; Adam in `gnn_models.py` | Explicit + code default |
| weight decay | **5e-4** | `gnn_models.py` / `default.yaml` (not in master YAML graphsage block) | Code / general config |
| λ (`campaign_loss_weight`) | **0.25** | `train_graphsage_fleet_correlation(..., campaign_loss_weight=0.25)` in `src/models/gnn_models.py` | Code default used by fleet correlator training |
| PCA dims | **8** | `campaign_clustering.run_dbscan(..., pca_components=8)`; `default.yaml` `dbscan_pca_components` | Code default / general config; used with publication DBSCAN |
| DBSCAN metric | **euclidean** | `DBSCAN(..., metric="euclidean")` in `src/evaluation/campaign_clustering.py` | Hardcoded in clustering helper |
| DBSCAN eps | **0.5** | `final_shared_fleet_configuration.yaml` `dbscan_eps` | Explicit freeze (**differs from default.yaml 1.2**) |
| DBSCAN min_samples | **2** | Freeze `dbscan_min_samples` | Explicit freeze (**differs from default.yaml 10**; **not η**) |
| β (`minimum_campaign_cohesion`) | **0.5** | Freeze `minimum_campaign_cohesion` | Explicit freeze |
| γ (`minimum_distinct_vehicles`) | **2** | Freeze `minimum_distinct_vehicles` | Explicit freeze |
| `minimum_cross_vehicle_support` | **1** | Freeze | Explicit freeze |
| η (`\|C_k\| >= η`) | **unresolved in shared-fleet freeze YAML** | Freeze file does not name η; paper definition preserved; DBSCAN `min_samples` must not be equated to η | Unresolved as freeze field |

---

## 2. Defaults vs publication (Task 2)

| Parameter | Publication value | General/default (`configs/default.yaml`) | Notes |
|-----------|-------------------|------------------------------------------|-------|
| τ | 0.95 | 0.95 | Same in current default graph block |
| k_same | 2 | 2 | Same |
| k_cross | 5 | 5 | Same |
| GraphSAGE hidden / emb / epochs / lr / wd | 64 / 32 / 30 / 0.01 / 5e-4 | Same under `gnn:` | Aligns |
| PCA | 8 | 8 | Aligns |
| **DBSCAN eps** | **0.5** | **1.2** | **Conflict — do not treat default as paper** |
| **DBSCAN min_samples** | **2** | **10** | **Conflict — do not treat default as paper** |
| clustering `similarity_threshold` (legacy cluster helper) | publication cohesion β=0.5 in freeze | 0.85 in `clustering:` / `final_decision:` | Different keys/contexts; freeze β is authoritative for publication gate |

`configs/fleet_ids.yaml` also carries several experiment-specific DBSCAN values (including 1.2 and 0.8 in places). Those are deliverables/auxiliary configs, not the shared-fleet freeze.

---

## 3. Parameter-selection provenance (Task 4)

Classification key:

- **A** — explicitly selected using training/validation experiments  
- **B** — selected through exploratory/preliminary experiments  
- **C** — fixed design choice  
- **D** — publication value known, but selection procedure cannot be recovered  

Artifact used:  
`…/validation_scenarios/parameter_search.csv` + `selection_report.md`  
(`candidates_tested: 14`, `search_stage` includes `dbscan` / `connectivity` / `campaign_rules`; objective uses V0–V4 validation metrics).  
**Limitation:** implementing module `joint_parameter_search` is listed in `MISSING_MODULES.md` (never committed), so the search cannot be re-executed from source even though the archived selection table exists.

| Parameter | Publication value | Provenance class | Notes |
|-----------|-------------------|------------------|-------|
| τ = 0.95 | 0.95 | **A** (archived validation selection) | Appears among searched thresholds `{0.85,0.9,0.95}` in `parameter_search.csv`; freeze `best_diag` selects 0.95 under constraints. Search **source code** missing. |
| k_same = 2 | 2 | **D** | Value confirmed in freeze; **never varied** in archived `parameter_search.csv` rows. Repository confirms the publication value but does not provide sufficient evidence to determine how this value was originally selected. |
| k_cross = 5 | 5 | **A** (archived validation selection) | Searched values include `{5,10}`; freeze selects 5 among constraint-satisfying rows. Search source missing. |
| DBSCAN eps = 0.5 | 0.5 | **A** (archived validation selection) | Searched `{0.5,0.8}`; `search_stage: dbscan`; best constraint-satisfying row uses 0.5. Higher raw objective rows with eps=0.8 fail constraints (`constraints_met: 0`). Search source missing. |
| DBSCAN min_samples = 2 | 2 | **A** (archived validation selection) | Searched `{2,3,4}`; freeze selects 2. **Not η.** Search source missing. |
| GraphSAGE dims / layers / Adam / λ=0.25 | as above | **D** / partial **C** | Architecture and λ appear as code/master defaults; no archived validation sweep table found that selects λ or layer count. |
| PCA = 8 | 8 | **D** | Used by clustering helpers; no dedicated selection artifact found for choosing 8. |
| β = 0.5, γ = 2 | 0.5 / 2 | **A** (campaign_rules stage in archive) / **D** for deeper motivation | Present in freeze; `parameter_search.csv` includes `campaign_rules` rows varying `minimum_cross_vehicle_support`. Full rule-search code missing. |
| η | unresolved in freeze YAML | **D** | Paper definition `\(\lvert C_k\rvert \ge \eta\)` is preserved; numeric field not recorded under that name in `final_shared_fleet_configuration.yaml`. |

When selection provenance cannot be established beyond the recorded value, the required statement applies:

> The repository confirms the parameter value used in the publication experiment, but does not provide sufficient evidence to determine how this value was originally selected.

---

## 4. Implementation facts (Task 5) — terminology & code references

Paths refer to the current `main`-line tree unless noted as recovered.

### 4.1 \(d_i\) — 24-D behavioural descriptor

- Defined as `BEHAVIOURAL_FEATURE_COLUMNS` in `src/features/feature_extractor.py` (24 columns: frame/CAN-ID/timing/DLC/byte stats).  
- Graph similarity feature view `full_descriptor` resolves these behavioural columns via `src/graph/fleet_graph_builder.py` / `fleet_similarity_features.py`.  
- **Used for behavioural similarity / graph construction**, not as the GraphSAGE input vector.

### 4.2 \(g_i\) — 9-D GraphSAGE node-feature vector

- Recovered publication feature schema (`GNN_FEATURE_COLUMNS` in recovered `final_gnn_fleet_decision_experiment.py`):  
  `anomaly_score`, `message_rate`, `frame_count`, `burstiness`, `mean_inter_arrival_time`, `std_inter_arrival_time`, `can_id_entropy`, `most_common_can_id_ratio`, `payload_entropy`.  
- On `main`, `src/graph/scenario_graph.py` documents that the original `prepare_gnn_fleet_node_matrix` path depends on modules not fully present on all branches.

### 4.3 \(\rho_{ij}\) and \(\omega_{ij}\)

- Graph construction uses **cosine** similarity (`similarity_metric: cosine`).  
- Retained edge weights are the cosine similarities after thresholding (`build_cross_vehicle_constrained_knn_edges` keeps `sim >= similarity_threshold` and stores those scores as weights).  
- Therefore for retained edges, stored weight equals \(\rho_{ij}\) (i.e. \(\omega_{ij} = \rho_{ij}\) for edges kept in the graph).

### 4.4 Edge weights vs SAGEConv

- `build_pyg_data` in `src/graph/fleet_graph_builder.py` stores `edge_attr=ew`.  
- `GraphSAGEFleetCorrelator.forward` / training loops call `model(x, edge_index)` — **no `edge_attr` argument** (`src/models/gnn_models.py`).  
- GraphSAGE therefore uses **graph connectivity via `edge_index`**; stored weights are not directly consumed by `SAGEConv` in the current implementation.

### 4.5 Behavioural cohesion for campaign decision

- Recovered `compute_cluster_behavioral_cohesion` computes mean cosine similarity to the centroid in **behaviour descriptor space** (`behavior_features`), explicitly “same space as graph edges”, not from stored `edge_attr` weights.

### 4.6 Pairwise similarity vs neighbor caps

- Publication graph path: `build_cross_vehicle_constrained_knn_edges` (`src/graph/fleet_graph_builder.py`) uses `sklearn.neighbors.NearestNeighbors` separately for same-vehicle and cross-vehicle candidates, then prunes with \(\tau\).  
- This is **constrained k-NN + threshold**, not an explicit “materialize full \(n\times n\) matrix then sparsify” API.  
- With the brute-force neighbor backend used for modest blocks, distance evaluation still scales with pairwise comparisons inside each query/fit set; **`k_same` / `k_cross` limit retained neighborhood size** and therefore graph density, but they do **not** remove the need for neighbor-search distance computation.

---

## 5. Reproducibility check (Task 6)

| Item | Status |
|------|--------|
| Publication runner | `recovered_publication_pipeline/scripts/run_balanced_publication_run.py` + balanced `runner.py` (on recovery branch / recovered tree; not a complete clean-`main` runnable package) |
| Config files | Freeze + master YAML in recovered tree; `configs/publication_fleet_guard.yaml` records them for `main` |
| Imports | Balanced runner imports missing packages listed in `MISSING_MODULES.md` (`joint_parameter_search`, historical `build_mixed_validation_suite`, `run_refinement_fcgnn`, `SharedFleetConfiguration`, etc.) |
| Required data | External OCSLab traces; balanced descriptors/models/intermediates not fully committed |
| Archived results | Present under `experimental-2026-06-23/01_primary_ocslab_balanced/` (tables/metrics) |
| Clean-checkout end-to-end reproduction of the original balanced publication run | **Not currently possible** without recovering missing modules and external data |

### Known Reproducibility Limitations

1. Multiple publication orchestration modules were **never committed** (`MISSING_MODULES.md`).  
2. Raw OCSLab dataset paths are external; descriptors/models for the balanced run are not fully in git.  
3. `configs/default.yaml` DBSCAN defaults (**eps=1.2, min_samples=10**) differ from the publication freeze (**0.5 / 2**); using default.yaml alone will not reproduce publication clustering settings.  
4. η is defined in the paper as \(\lvert C_k\rvert \ge \eta\) but is **not named** in `final_shared_fleet_configuration.yaml`; do not equate η with DBSCAN `min_samples`.  
5. This cleanup **documents** the publication configuration; it does **not** repair missing packages or rewrite the pipeline.

---

## 6. Confidence summary

| Area | Confidence |
|------|------------|
| Graph τ / k_same / k_cross publication values | **High** (freeze + archives) |
| Publication DBSCAN eps/min_samples | **High** (freeze) |
| β / γ freeze values | **High** |
| GraphSAGE 2×SAGEConv, ReLU, Adam, dims, λ=0.25 | **High** for architecture/code defaults; master YAML omits some fields |
| 9-D GNN input schema | **High** in recovered GNN module; prep wiring incomplete on some branches |
| η numeric freeze field | **Low / unresolved** in shared-fleet YAML |
| Full re-execution from clean checkout | **Blocked** (missing modules + data) |

---

## 7. What this audit did not do

- No algorithm / model / experiment-result changes  
- No archived output modifications  
- No retuning, reruns, TEST/CTT execution  
- No invented “parameters were selected on validation” claims beyond what `parameter_search.csv` / `selection_report.md` actually record  
