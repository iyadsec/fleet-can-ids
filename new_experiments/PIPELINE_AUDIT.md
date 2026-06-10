# Pipeline Audit — Fleet-Aware CAN IDS

**Audit date:** 2026-06-05  
**Repository:** `/Users/iyadatieh/Library/CloudStorage/OneDrive-Personal/University of Reading/CodeRepo/Code`  
**Purpose:** Map existing pipeline stages before implementing controlled scenario experiments under `new_experiments/`.

---

## Executive summary

The repository implements a multi-stage fleet IDS pipeline: benign-only Isolation Forest per vehicle → anomaly descriptors → behavioural-similarity graph (no temporal edges) → GraphSAGE message passing → DBSCAN on embeddings → isolated vs coordinated campaign decision.

**Two parallel fleet paths exist:**

| Path | Entry | Clustering space | Decision |
|------|-------|------------------|----------|
| **IEEE / H4 (primary)** | `run_final_gnn_fleet_decision_evaluation.py` | DBSCAN on **GNN embeddings** | `isolated_attack` / `coordinated_attack` |
| **Legacy descriptor** | `run_campaign_detection_evaluation.py` | DBSCAN on **descriptors** | Campaign match vs ground truth |
| **Legacy full pipeline** | `experiments/run_full_pipeline.py` | DBSCAN on GNN embeddings (descriptor fallback) | `Isolated anomaly` / `Fleet-level coordinated…` |

**Naming note:** Documentation and the new experiment spec refer to **FCGNN** (fleet campaign GNN). The codebase implements this as `GraphSAGEFleetCorrelator` in `src/models/gnn_models.py` — a two-layer GraphSAGE encoder with structure-based supervision. There is no separate `FCGNN` class.

---

## 1. Pipeline stages — file and function map

### 1.1 Dataset loading

| Stage | File | Function(s) | Reuse |
|-------|------|-------------|-------|
| Primary Car Track loader | `src/data/dataset_loader.py` | `discover_log_files`, `load_single_file`, `iter_load_files`, `load_and_merge`, `save_clean_dataset` | **Reuse as-is** |
| Release dataset loader | `src/data/release_dataset_loader.py` | `discover_release_files`, `consolidate_release_dataset`, `load_release_file` | Reuse for release logs |
| Stub loader | `src/data/loaders.py` | `load_can_trace` | **Not implemented** — ignore |
| Preprocessing placeholder | `src/data/preprocessing.py` | `preprocess_trace` | Minor only |
| Validation | `src/data/validate_dataset.py`, `src/data/stage1_consolidated_validation.py` | validation helpers | Reuse for QA |
| Experiment wrappers | `experiments/01_load_dataset.py`, `experiments/01b_validate_clean_dataset.py` | CLI | Reuse |

**Config:** `configs/fleet_ids.yaml` → `data.external_dataset_dir`, `paths.raw_dir`

### 1.2 Window generation

| File | Function(s) | Notes |
|------|-------------|-------|
| `src/features/window_generator.py` | `resolve_window_params`, `generate_windows_for_trace`, `generate_windows`, `load_can_frames` | Sliding windows per `source_file`; default size=100, overlap=50, stride=50 |
| `experiments/02_generate_windows.py` | CLI wrapper | |

### 1.3 Isolation Forest training and inference

| File | Function(s) | Notes |
|------|-------------|-------|
| `src/models/vehicle_ids.py` | `fit_self_supervised_isolation_forest`, `score_self_supervised_isolation_forest`, `_normalised_anomaly_percentile`, `generate_vehicle_anomaly_predictions`, `save_vehicle_ids_model`, `load_vehicle_ids_model` | Benign-only fit per vehicle; scores all windows |
| `src/evaluation/vehicle_level_evaluation.py` | `run_vehicle_level_evaluation` | Paper Table 1, ROC, threshold sweep with explicit splits |
| `experiments/04_train_vehicle_ids.py` | CLI | Writes `outputs/models/vehicle_isolation_forest.joblib` |

### 1.4 Anomaly score normalization and thresholds

| File | Function(s) | Logic |
|------|-------------|-------|
| `src/models/vehicle_ids.py` | `_normalised_anomaly_percentile` | Rank-based map to [0,1] vs benign reference |
| `src/models/vehicle_ids.py` | `generate_vehicle_anomaly_predictions` | `local_alert = score ≥ strong_threshold`; `weak_signal = weak ≤ score < strong` |
| Config | `configs/fleet_ids.yaml` → `vehicle_ids` | `strong_threshold: 0.80`, `weak_threshold: 0.55` |

Evidence levels: `strong_local_anomaly` | `weak_suspicious_signal` | `normal`.

### 1.5 Descriptor generation

| File | Function(s) | Output |
|------|-------------|--------|
| `src/features/descriptor_generator.py` | `generate_anomaly_descriptors`, `build_transmitted_descriptors`, `save_anomaly_descriptors` | `anomaly_descriptors.csv` |
| `src/features/feature_extractor.py` | `BEHAVIOURAL_FEATURE_COLUMNS`, `extract_features` | 22 behavioural features per window |
| `experiments/05_generate_descriptors.py` | CLI | |

Filter: retains rows with `evidence_level ∈ {strong_local_anomaly, weak_suspicious_signal}`.

### 1.6 Graph node construction

| File | Function(s) | Node definition |
|------|-------------|-----------------|
| `src/graph/fleet_graph_builder.py` | `load_anomaly_descriptors`, `parse_feature_matrix`, `build_networkx_graph`, `build_pyg_data`, `attach_ground_truth_labels` | **One node = one suspicious window descriptor** (`event_id`); attrs: `vehicle_model`, `anomaly_score`, `local_alert`, `weak_signal`, `attack_type` |
| `src/graph/fleet_graph.py` | `FleetGraph`, `build_fleet_graph` | **Stub** — not used in main pipeline |

### 1.7 Cosine similarity and kNN graph construction

| File | Function(s) | Mechanism |
|------|-------------|-----------|
| `src/graph/fleet_graph_builder.py` | `build_similarity_edges` | Radius NN on cosine |
| `src/graph/fleet_graph_builder.py` | `build_topk_similarity_edges` | Top-k + τ prune |
| `src/graph/fleet_graph_builder.py` | `build_cross_vehicle_constrained_knn_edges` | **IEEE path:** k_same=10, k_cross=5, cosine, τ=0.95 |
| `src/graph/fleet_similarity_features.py` | `prepare_fleet_similarity_matrix`, `build_behavior_view_descriptors` | Behaviour views for similarity |

**No temporal features in edge criteria** — explicit in `build_similarity_edges` comments.

### 1.8 Cross-vehicle edge handling

| File | Function(s) |
|------|-------------|
| `src/graph/fleet_graph_builder.py` | `build_cross_vehicle_constrained_knn_edges`, `build_cross_vehicle_constrained_graph`, `build_cross_vehicle_similarity_edges`, `merge_edge_sets`, `graph_to_tables` (`is_cross_vehicle_edge`) |

### 1.9 GNN implementations

| File | Class / function | Role |
|------|------------------|------|
| `src/models/gnn_models.py` | `GraphSAGEEncoder`, `GCNEncoder`, `GATEncoder` | Standard baselines |
| `src/models/gnn_models.py` | `GraphSAGEFleetCorrelator` | **Proposed fleet correlator (M4 / “FCGNN”)** |
| `src/models/gnn_models.py` | `train_graphsage_fleet_correlation` | Structure supervision (link loss + anomaly MSE) |
| `src/evaluation/final_gnn_fleet_decision_experiment.py` | `run_gnn_fleet_correlation`, `build_final_gnn_fleet_graph` | IEEE orchestration |

### 1.10 Embedding extraction

| File | Function(s) |
|------|-------------|
| `src/models/gnn_models.py` | `train_graphsage_fleet_correlation`, `save_node_embeddings` |
| `src/evaluation/campaign_clustering.py` | `load_embedding_table` |
| `src/evaluation/final_gnn_fleet_decision_experiment.py` | `run_gnn_fleet_correlation` → `final_gnn_node_embeddings.csv` |

### 1.11 DBSCAN / clustering

| File | Function(s) | Space |
|------|-------------|-------|
| `src/evaluation/campaign_clustering.py` | `run_dbscan`, `extend_dbscan_labels`, `DbscanProjector` | Scaled PCA → Euclidean DBSCAN |
| `src/evaluation/campaign_detection_experiment.py` | `detect_campaign_clusters` | **Descriptors** |
| `src/evaluation/final_gnn_fleet_decision_experiment.py` | `cluster_gnn_embeddings` | **GNN embeddings** |

Defaults: eps=1.2 (most scripts), **0.8** in `final_gnn_fleet_decision`; min_samples=10; PCA=8.

### 1.12 Campaign decision logic

**IEEE path** (`src/evaluation/final_gnn_fleet_decision_experiment.py`):

| Function | Rule |
|----------|------|
| `cluster_gnn_embeddings` | Qualifying cluster: `size ≥ min_cluster_size`, `vehicles ≥ min_vehicles`, `behavioral_cohesion ≥ 0.85` |
| `assign_final_decisions` | `coordinated_attack` if node’s DBSCAN cluster is qualifying; else `isolated_attack` |
| `compute_cluster_behavioral_cohesion` | Mean cosine to centroid in **behaviour feature space** |

**Legacy path** (`src/evaluation/final_decision.py`): coordinated if multi-vehicle + mean similarity threshold.

**Legacy campaign detection** (`campaign_detection_experiment.py`): uses `min_dominant_attack_ratio` (0.60) — attack metadata in cluster validity.

### 1.13 Metric generation

| File | Role |
|------|------|
| `src/evaluation/metrics.py` | Generic sklearn metrics |
| `src/evaluation/final_gnn_fleet_decision_experiment.py` | `evaluate_local_vs_gnn`, campaign tables |
| `src/evaluation/vehicle_level_evaluation.py` | H1 vehicle-level metrics |
| `src/evaluation/ieee_paper_exports.py` | Paper bundle |

### 1.14 Runtime / memory utilities

| Location | Mechanism |
|----------|-----------|
| `src/pipeline/full_pipeline.py` | `time.perf_counter()` per step |
| `src/evaluation/campaign_clustering.py` | `subsample_indices` for large DBSCAN |
| `src/graph/fleet_graph_builder.py` | `max_nodes` subsampling |

**Gap:** No dedicated `psutil` / `tracemalloc` module — **new experiment must add** `src/experiments/runtime_measurement.py`.

---

## 2. Reuse vs modification matrix

| Component | Reuse as-is | Needs modification / new wrapper |
|-----------|-------------|-----------------------------------|
| Data loading | `dataset_loader.load_and_merge` | Scenario subsampling in `scenario_generator.py` |
| Windows / features | `window_generator`, `feature_extractor` | Trace-grouped splits for leakage prevention |
| Vehicle IDS | `generate_vehicle_anomaly_predictions` | Enforce benign-only train split; wire `test_size` |
| Descriptors | `generate_anomaly_descriptors` | Coordination-strength dial in `coordination_strength.py` |
| Graph builder | `build_cross_vehicle_constrained_knn_edges` | Parameterised τ and k for edge sensitivity |
| Descriptor clustering (M2) | `detect_campaign_clusters` | Align gates with IEEE (remove attack-ratio gate option) |
| Standard GNN (M3) | `GCNEncoder` / `train_gnn` | Match embedding dim, epochs, structure supervision |
| Proposed GNN (M4) | `GraphSAGEFleetCorrelator` | Wrap in `method_fcgnn.py` |
| Campaign decision | `assign_final_decisions`, `cluster_gnn_embeddings` | Reuse gates; export weak events |
| Metrics | `metrics.py` | Extend in `campaign_evaluation.py` |
| Statistics | — | **New** `statistical_testing.py` |
| Output safety | — | **New** `result_writer.py` |

---

## 3. Audit questions (required answers)

### 3.1 Random Forest references

Present **only** in privacy / generalisation experiments — **not** in IDS, graph, GNN, or campaign decision:

- `src/evaluation/cross_vehicle_generalisation_experiment.py` — `RandomForestClassifier`
- `src/evaluation/descriptor_security_experiment.py` — `RandomForestRegressor`
- `src/evaluation/privacy_evidence.py` — `RandomForestRegressor`

**Vehicle IDS remains Isolation Forest only.**

### 3.2 Temporal graph edges

**None.** All edges are behavioural cosine similarity between descriptor vectors. Inter-arrival features exist in node features but are **not** used for adjacency.

### 3.3 Weak anomalies retained for fleet processing?

**Yes.** Descriptors include `weak_suspicious_signal`; graph nodes include weak events; clustering runs on all descriptor nodes.

**Caveat:** `final_attack_decisions.csv` exports only `local_alert == 1` — weak-only events are clustered but omitted from primary decision export. New experiments must report weak-event recovery explicitly.

### 3.4 Is GNN output used in final campaign decision?

| Signal | Used? |
|--------|-------|
| GNN **embeddings** → DBSCAN cluster ID | **Yes** |
| GNN **`campaign_score`** | **No** — `campaign_score_threshold` (0.55) defined in config but **not applied** in `assign_final_decisions` |
| Behavioural cohesion gate | **Yes** |

### 3.5 Could descriptor clustering alone produce the same decision?

**Largely yes** via `run_campaign_detection_evaluation.py` → `detect_campaign_clusters` on descriptor features with the same graph and similar DBSCAN/cohesion gates.

Differences from IEEE GNN path:

- Clustering space: descriptors vs GNN embedding space
- Legacy uses `min_dominant_attack_ratio`; IEEE uses behaviour cohesion only
- GNN node features use narrower `GNN_FEATURE_COLUMNS` with per-vehicle z-scoring

**M2 ablation must use IEEE-equivalent gates** (no attack-ratio gate) for fair comparison.

### 3.6 PyTorch Geometric bidirectional edges

NetworkX stores undirected edges once. PyG `edge_index` duplicates each edge in both directions:

```python
# src/graph/fleet_graph_builder.py (build_pyg_data)
edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
edge_weights = np.concatenate([w, w])
```

**Unique undirected relationships ≠ PyG stored edge count** (PyG count ≈ 2× unique).

### 3.7 Campaign ground truth representation

`src/evaluation/campaign_detection_experiment.py` → `build_campaign_ground_truth`:

- One row per (campaign, vehicle, window)
- `campaign_id` = `CAMP-{attack_type}` when ≥2 vehicles share that attack type
- Columns: `campaign_id`, `attack_type`, `vehicle_id`, `window_id`, `descriptor_id`, `anomaly_score`

New scenarios (S0–S4) require **explicit scenario-level ground truth** beyond dataset-derived campaigns — to be built in `scenario_generator.py` without using model predictions.

### 3.8 Train/test leakage risks

| Risk | Location | Severity |
|------|----------|----------|
| IDS scores all windows | `generate_vehicle_anomaly_predictions` — `test_size` unused | **High** |
| GNN random node masks | `gnn_models._random_masks` — not trace holdout | Medium |
| Legacy GNN ground-truth labels | `gnn.use_ground_truth_labels: true` in config | High if legacy path used |
| IEEE mitigates | `prefer_ground_truth_labels=False`, `gnn_supervision: structure` | Lower for H4 |
| DBSCAN extend_labels | Transductive assignment to all nodes | Low–medium |
| Eval uses attack_type matching | `build_campaign_by_attack_table` | Eval only |
| **Bug** | `evaluate_local_vs_gnn` L424: GNN attack metrics copy local | **Misleading metrics** |

**New experiments must:** group overlapping windows by trace during splitting; hold out test traces for scenario generation; document splits in `scenario_membership.csv`.

---

## 4. Methodological inconsistencies (documented, not silently fixed)

1. **Dual graph τ values:** `graph.similarity_threshold: 0.85` vs `fleet_graph.similarity_threshold: 0.95` vs IEEE final path 0.95.
2. **GNN supervision mismatch:** `gnn.use_ground_truth_labels: true` vs `final_gnn_fleet_decision.gnn_supervision: structure`.
3. **FCGNN naming vs implementation:** Spec says FCGNN; code has `GraphSAGEFleetCorrelator`.
4. **Unused `campaign_score_threshold`** in final decision.
5. **Attack metadata in legacy cluster validity** vs behaviour-only IEEE gating.
6. **Weak signals:** included upstream, filtered from `final_attack_decisions.csv`.
7. **`test_size` dead parameter** in canonical IDS generator.
8. **GNN vs local attack metrics bug** in `evaluate_local_vs_gnn`.

Corrections for items 4–8 will be addressed within `new_experiments/` wrappers only; existing `results/`, `figures/`, `tables/`, `paper/` outputs remain untouched.

---

## 5. Key entry-point map

```
experiments/01_load_dataset.py              → dataset_loader
experiments/02_generate_windows.py        → window_generator
experiments/03_extract_features.py        → feature_extractor
experiments/04_train_vehicle_ids.py         → vehicle_ids
experiments/05_generate_descriptors.py      → descriptor_generator
experiments/06_build_graph.py               → fleet_graph_builder
experiments/07_train_gnn.py                 → gnn_models (legacy)
experiments/08_cluster_campaigns.py         → campaign_clustering
experiments/09_final_decision.py            → final_decision

run_final_gnn_fleet_decision_evaluation.py  → IEEE H4 (M4 prototype)
run_campaign_detection_evaluation.py        → descriptor DBSCAN (M2 prototype)
run_vehicle_level_evaluation.py             → H1 local IDS (M1 prototype)
run_cross_vehicle_generalisation.py         → H3 (not part of M1–M4 ablation)

scripts/run_new_scenario_experiments.py       → NEW controlled experiments
configs/fleet_ids.yaml                        → existing pipeline hyperparameters
new_experiments/configs/scenario_experiments.yaml → NEW experiment config
```

---

## 6. Phase 1 status

- [x] Pipeline audit documented
- [x] `new_experiments/` directory structure created
- [x] `scenario_experiments.yaml` configuration created
- [x] Safe output handling (`result_writer.py`) — writes only under `new_experiments/`
- [x] Scenario registry (S0–S4 definitions)
- [x] Master runner with `--dry-run`
- [ ] Scenario generator implementation (Phase 2+)
- [ ] Method implementations M1–M4 (Phase 2–6)
- [ ] Full experiment execution (Phase 9)
