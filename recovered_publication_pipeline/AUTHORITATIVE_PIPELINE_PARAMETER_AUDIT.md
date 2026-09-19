# Authoritative primary OCSLab publication pipeline — parameter audit

**Verdict:** The numbers Strong Campaign F1 `0.533 / 0.733 / 1.000`, Weak `0.067 / 0.500 / 0.717`, unrelated merge `0.400` come from the **balanced end-to-end publication run** on `origin/cursor/campaign-clustering` @ `61a82038ffcac2066d57d524f9305be88dabe212`, originally written under:

`new_experiments/final_end_to_end_publication_run_balanced/`

That tree was later copied (not re-run) into:

`experimental-2026-06-23/01_primary_ocslab_balanced/`

by `scripts/consolidate_experimental_results.py` (commit `e481af5`). Frozen reference copies also live under `recovered_publication_pipeline/frozen_results_reference/`.

**Master config hash (authoritative):** `72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e`

**Entry point:** `scripts/run_balanced_publication_run.py` → `src.experiments.final_end_to_end_publication_run_balanced.runner.run_balanced_publication`

**Per-scenario executor (MISSING source):** `run_refinement_fcgnn` from `src.experiments.coordinated_campaign_refinement.refinement_pipeline`, called indirectly via missing `_run_single_test`.

**Reproduction status:** **NOT REPRODUCIBLE** from git alone (`RECOVERY_REPORT.md` class C). Fifteen packages that the runner imports were never committed. Parameter values below are recovered from (1) frozen YAML/artifacts, (2) recovered peer modules that the runner wired to, (3) explicit overrides documented in freeze artifacts. Where only peer defaults exist and the freeze consumer is missing, that is marked **WIRING UNRECOVERED**.

---

## 1. Identity chain (do not invent alternate pipelines)

| Layer | Path / evidence |
|-------|-----------------|
| Canonical archive | `experimental-2026-06-23/01_primary_ocslab_balanced/` — `BALANCED_PUBLICATION_SUMMARY.md` lists the exact F1/merge numbers and hash |
| Original producer | `new_experiments/final_end_to_end_publication_run_balanced/` on `campaign-clustering` @ `61a82038` |
| Consolidation | `scripts/consolidate_experimental_results.py` `GIT_COPIES` maps `…/table_P7_*.csv` → `01_primary_ocslab_balanced/tables/…` |
| Provenance label | `experimental-2026-06-23/02_baseline_ablation/RESULT_PROVENANCE.md`: result set **B** = “Balanced publication (authoritative)” |
| Recovered snapshot | `recovered_publication_pipeline/` (exact historical copies; see `PROVENANCE.md`) |

**Not authoritative for these numbers:** framework ablation `table_06` (different F1), original (pre-balanced) split (strong cs=5 F1 was 0.867), current workspace `experiments/run_full_pipeline.py` / `docs/paper_pipeline.md` stage list (generic mapping, not the balanced runner), reviewer ablation harnesses (explicitly not bit-comparable).

---

## 2. Call graph (actual callers)

```
scripts/run_balanced_publication_run.py
  └─ final_end_to_end_publication_run_balanced.runner.run_balanced_publication
       ├─ _write_master_config / _load_master_config  → balanced_master_experiment.yaml
       ├─ retrain_local_pipeline                     [MISSING package]
       ├─ ensure_fleet_scaler_in_config              [RECOVERED]
       ├─ build_mixed_validation_suite               [MISSING]
       ├─ joint_parameter_search → SharedFleetConfiguration  [MISSING]
       │     writes final_shared_fleet_configuration.yaml   [RECOVERED freeze]
       ├─ enumerate_test_runs / PUBLICATION_SCENARIOS / REQUIRED_SEEDS  [MISSING]
       ├─ for each run: _run_single_test(..., shared=shared)  [MISSING]
       │     └─ (historically) run_refinement_fcgnn          [MISSING]
       ├─ generate_tables → table_P7 / table_P8              [MISSING base + RECOVERED wrapper]
       └─ _write_summary → BALANCED_PUBLICATION_SUMMARY.md
```

Evidence: `recovered_publication_pipeline/src/experiments/final_end_to_end_publication_run_balanced/runner.py` imports and call sites (lines 16–47, 264–477).

---

## 3. Parameter audit

Evidence grades:

- **FREEZE** — present in recovered balanced YAML/CSV/JSON from the publication run
- **CODE** — present in recovered peer source that the balanced runner imported or that defines the same named components
- **WIRING UNRECOVERED** — value known from freeze or peer default, but the missing package that applied it is not in git
- **NOT FOUND** — no evidence in recoverable artifacts

### 3.1 Descriptor dimensions

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Stored behavioural descriptor schema | `BEHAVIOURAL_FEATURE_COLUMNS` — **24 features** in this exact order: `frame_count`, `unique_can_id_count`, `can_id_entropy`, `most_common_can_id_ratio`, `mean_inter_arrival_time`, `std_inter_arrival_time`, `mean_dlc`, `std_dlc`, `byte_mean_0`…`byte_mean_7`, `byte_std_0`…`byte_std_7` | **CODE** `recovered_publication_pipeline/src/features/feature_extractor.py` `BEHAVIOURAL_FEATURE_COLUMNS`; master YAML `descriptor.feature_schema: BEHAVIOURAL_FEATURE_COLUMNS` (**FREEZE** `configs/balanced_master_experiment.yaml`) |
| Privacy exclusions | `can_id`, `payload_bytes`, `raw_timestamp` | **FREEZE** `descriptor.privacy_excluded` |
| Note | Older `PIPELINE_AUDIT.md` says “22” features — that conflicts with the recovered list (24). Prefer the source list. | Ambiguity |

### 3.2 GraphSAGE / fleet node input features (exact names & order)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| GNN input feature vector (9-D) | Exact order in `GNN_FEATURE_COLUMNS`: **(1)** `anomaly_score`, **(2)** `message_rate`, **(3)** `frame_count`, **(4)** `burstiness`, **(5)** `mean_inter_arrival_time`, **(6)** `std_inter_arrival_time`, **(7)** `can_id_entropy`, **(8)** `most_common_can_id_ratio`, **(9)** `payload_entropy` | **CODE** `recovered_publication_pipeline/src/evaluation/final_gnn_fleet_decision_experiment.py` `GNN_FEATURE_COLUMNS` + `prepare_gnn_fleet_node_matrix` |
| Derived features | `message_rate` ← `frame_count`; `burstiness` ← `std_inter_arrival_time / (|mean_inter_arrival_time|+1e-9)`; `payload_entropy` ← entropy of abs(`byte_mean_*`) row-normalized | **CODE** `fleet_similarity_features.build_behavior_view_descriptors`, `compute_payload_entropy` |
| Local normalisation before GNN | Benign-train z-score via `apply_fleet_scaler` on those 9 names | **CODE** `prepare_gnn_fleet_node_matrix`; **FREEZE** scaler JSON + `scalers/scaler_manifest.csv` (`fitted_feature_names` match the 9 names; `fit_row_count=11423`, benign train only) |
| Input dim statement | 9 (not encoded in master YAML; code-level) | **CODE** / **CONFIG_COMPARISON.md** |

### 3.3 Graph construction

| Parameter | Value | Evidence it was used for balanced publication |
|-----------|-------|-----------------------------------------------|
| τ (`similarity_threshold`) | **0.95** | **FREEZE** `final_shared_fleet_configuration.yaml`; table P3; audit JSON in `original_vs_balanced_split.md`; selection_report |
| `k_same` / `max_same_vehicle_neighbors` | **2** | **FREEZE** same files (balanced freeze; original non-balanced had cross=10 but same=2) |
| `k_cross` / `max_cross_vehicle_neighbors` | **5** | **FREEZE** same |
| Metric / cosine path | Cosine k-NN with same/cross caps, then prune edges with `sim < τ` | **CODE** `fleet_graph_builder.build_cross_vehicle_constrained_knn_edges` (`metric="cosine"` only); called from `build_final_gnn_fleet_graph` |
| Feature space for edges | Same locally normalised behaviour view used for GNN node matrix (via `prepare_gnn_fleet_node_matrix` → `X`) | **CODE** `final_gnn_fleet_decision_experiment.build_final_gnn_fleet_graph` |
| Temporal edges | **false** | **FREEZE** `fleet_graph.temporal_edges: false` in master YAML |
| Scaler for similarity / GNN | Fleet benign StandardScaler-equivalent (means/stds JSON), **not** fit on scenario nodes | **FREEZE** `scalers/fleet_benign_scaler.json`; **CODE** `local_descriptor_normalisation.apply_fleet_scaler` |
| Edge weights stored | Cosine similarities stored as `edge_attr` on PyG `Data` | **CODE** `build_pyg_data` sets `edge_attr=ew` |
| Edge weights in GraphSAGE message passing | **Not used** — `GraphSAGEFleetCorrelator.forward(x, edge_index)` / `SAGEConv` calls ignore `edge_attr` | **CODE** `gnn_models.py` |
| Default vs freeze for k_same | Peer `FinalGnnFleetConfig.top_k_same_vehicle` **default 10**; publication freeze is **2** | Freeze must be applied by missing `SharedFleetConfiguration` consumer → **WIRING UNRECOVERED** |

### 3.4 GraphSAGE architecture

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Class | `GraphSAGEFleetCorrelator` (documented as FCGNN / method label `fcgnn` in metrics) | **CODE** `gnn_models.py`; frozen `campaign_metrics.csv` column `method=fcgnn` |
| Layers / dims | **9 → 64 → 32** (`SAGEConv(in,64)`, `SAGEConv(64,32)` + linear classifier + campaign scorer) | **FREEZE** `graphsage.hidden_dim: 64`; **CODE** `embedding_dim=32` default in `train_graphsage_fleet_correlation` / `FinalGnnFleetConfig.gnn_embedding_dim` |
| Aggregation | PyG `SAGEConv` default (**mean**) — constructor does not pass `aggr=` | **CODE** `GraphSAGEFleetCorrelator.__init__` |
| Activation | **ReLU** after first SAGEConv; second layer linear embedding | **CODE** `F.relu(self.conv1(...)); z = self.conv2(h, ...)` |
| Campaign head | `Linear(32→1)+Sigmoid` | **CODE** |

### 3.5 Training

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Objective | Structure supervision: link reconstruction on edges + MSE of campaign score to min-max scaled `anomaly_score` feature; `loss = link_loss + λ * camp_loss` | **CODE** `train_graphsage_fleet_correlation` (`supervision="structure"` default; `FinalGnnFleetConfig.gnn_supervision`) |
| λ (`campaign_loss_weight`) | **0.25** | **CODE** default in `train_graphsage_fleet_correlation`; **not** in master YAML → **WIRING UNRECOVERED** for whether missing `run_refinement_fcgnn` overrode it (no freeze key exists; peer default is the only evidence) |
| Optimizer | Adam | **CODE** |
| Learning rate | **0.01** | **FREEZE** `graphsage.learning_rate`; **CODE** default |
| Weight decay | **5e-4** | **CODE** `train_graphsage_fleet_correlation` / `FinalGnnFleetConfig.gnn_weight_decay`; **not** in master YAML → **WIRING UNRECOVERED** (same caveat as λ) |
| Epochs | **30** | **FREEZE** `graphsage.epochs`; **CODE** |
| Train/val node mask ratios | 0.7 / 0.15 (random node masks, not trace holdout) | **CODE** defaults |
| Labels in structure mode | Attack labels **not** used for the structure objective; `prefer_ground_truth_labels=False` on PyG build in IEEE path | **CODE** `build_final_gnn_fleet_graph` |

### 3.6 Embedding clustering

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Method | DBSCAN | **FREEZE** `clustering.method`; shared config |
| StandardScaler before PCA | **Yes** (`DbscanProjector.scaler = StandardScaler()`) | **CODE** `campaign_clustering.DbscanProjector` / `run_dbscan` |
| PCA dims | **8** | **CODE** `FinalGnnFleetConfig.dbscan_pca_components=8`, `run_dbscan(pca_components=8)`; not in master YAML |
| DBSCAN metric | **euclidean** (on PCA space) | **CODE** `DBSCAN(..., metric="euclidean")` |
| eps | **0.5** (publication) | **FREEZE** `dbscan_eps: 0.5` |
| min_samples | **2** (publication) | **FREEZE** `dbscan_min_samples: 2` |
| Code defaults vs freeze | `FinalGnnFleetConfig` still defaults **eps=1.2, min_samples=10** | Freeze application requires missing `SharedFleetConfiguration` → **WIRING UNRECOVERED** |
| Extend labels | Nearest PCA centroid within `eps`, else noise | **CODE** `extend_dbscan_labels` |

### 3.7 Campaign gate (γ, η, β) and cohesion

Frozen shared gate (`final_shared_fleet_configuration.yaml` / table P3):

| Freeze key | Value | Greek symbol in repo? |
|------------|-------|------------------------|
| `minimum_distinct_vehicles` | **2** | **NOT FOUND** as γ in authoritative balanced artifacts |
| `minimum_cross_vehicle_support` | **1** | **NOT FOUND** as η in authoritative balanced artifacts |
| `minimum_campaign_cohesion` | **0.5** | Explicitly called **β** in later ablation docs that cite the publication pair |
| `fragment_centroid_threshold` | **0.85** | Distinct from β; fragment merge |
| `fragment_consolidation_enabled` | **true** | FREEZE |

**β evidence:** `experimental-reviewer-ablation/real_ocslab/provenance.md` / `config.yaml` label `cohesion_threshold` (β) = 0.5 and state this is the publication pair (0.5 cohesion / 0.85 fragment). Those docs are **not** the balanced runner itself, but they intentionally cite the publication freeze.

**γ / η:** No recoverable authoritative source assigns the symbols γ/η to freeze keys. Closest named freeze fields are `minimum_distinct_vehicles=2` and `minimum_cross_vehicle_support=1`. Do not treat greek mapping as proven.

**Cohesion formula (two recovered peers — which one `run_refinement_fcgnn` used is UNRECOVERED):**

1. **Centroid cohesion (IEEE GNN decision path):** mean cosine similarity of L2-normalised behaviour features to the cluster centroid — `compute_cluster_behavioral_cohesion` in `final_gnn_fleet_decision_experiment.py`.
2. **Mean pairwise cohesion (summarize_clusters path):** mean upper-triangle pairwise cosine — `mean_intra_cluster_similarity` in `campaign_clustering.py`.

Publication freeze cohesion threshold **0.5** is lower than IEEE default `min_behavioral_cohesion=0.85` in `FinalGnnFleetConfig`.

**Fragment consolidation (peer approximation, not publication source):** ablation `apply_campaign_gate` merges accepted campaigns whose **embedding** centroids have cosine ≥ `fragment_merge_threshold` (0.85). Publication freeze names this `fragment_centroid_threshold`. Exact implementation lived in missing modules.

### 3.8 Matching logic

| Aspect | Evidence |
|--------|----------|
| Publication metric extraction | `extract_run_metrics` / `safety_row` in **MISSING** `final_shared_configuration.metrics` |
| Frozen outputs | `campaign_metrics.csv` columns include `campaign_precision/recall/f1`, `membership_*`, `incorrect_merging`, `false_campaign_rate`, etc. |
| Closest documented matcher (ablation, **not** proven identical) | Greedy Jaccard ≥ 0.5 between predicted clusters and GT campaigns — `experimental-reviewer-ablation/metrics.py` `match_campaigns`; ablation provenance says historical publication matching modules were never committed |
| Unrelated merge 0.400 | **FREEZE** `table_P6_benign_isolated_unrelated_results.csv` row `Unrelated Multi-Vehicle Incidents` → `incorrect_merging=0.4` |

**Ambiguity:** Exact campaign↔GT matching used to produce table_P7/P8 is unrecovered.

### 3.9 Seeds

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Scenario / statistics seeds | **11, 23, 37, 41, 53, 67, 71, 83, 97, 101** | **FREEZE** `local_ids.random_seeds` and `statistics.seeds` in master YAML; **CODE** `campaign_analysis_corrected.REQUIRED_SEEDS` (same list); per-seed rows in `campaign_metrics.csv` |
| Aggregate `seed` column in P7/P8 | **58.4** = mean of the ten seeds | Arithmetic check on table_P7 |
| IF retrain seed | **42** (runner hardcode when calling `retrain_local_pipeline`) | **CODE** `runner.py` `seed=42` — local IDS only |
| GNN per-run seed | Scenario seed passed into missing `_run_single_test` / refinement | **WIRING UNRECOVERED** |

### 3.10 Node budget / scenario geometry

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Fleet size | **20** vehicles | **FREEZE** `scenario.fleet_size`; table P7 `fleet_size=20` |
| Windows / vehicle | **10** | **FREEZE** `source_windows_per_vehicle` |
| Total nodes | **200** | **FREEZE** `total_source_windows`; table P7 `total_nodes=200` |
| DescriptorBudget | `descriptors_per_vehicle=10`, `malicious_per_attacked=5`, `benign_per_attacked=5`, `benign_per_benign=10`, `total_fleet_size=20` | **CODE** constructed in `runner.run_balanced_publication` |
| Campaign sizes | **2, 5, 10** | **FREEZE** `scenario.campaign_sizes` |
| Strong/weak thresholds | weak **0.55**, strong **0.80** | **FREEZE** `local_ids.*` |
| Behavioural coordination only | **true** | **FREEZE** `scenario.behavioural_coordination_only` |

### 3.11 Local IDS / windowing (upstream of fleet)

| Parameter | Value | Evidence |
|-----------|-------|----------|
| Window size / stride / overlap | 100 / 50 / 50 | **FREEZE** `windowing.*`; guard gap 100 frames |
| IsolationForest | `n_estimators=200`, `contamination=auto`, threshold grid on validation F1 | **FREEZE** `local_ids.*` |
| Train ratios | 0.70 / 0.15 / 0.15; benign-only training; source-level separation | **FREEZE** `dataset.*` |
| Platforms | Hyundai, Kia, Chevrolet (balanced segments for Chevy) | **FREEZE** + `platform_split_summary.csv` |

---

## 4. How table_P7 / table_P8 were produced

1. Per-seed runs written to `results/scenario_evaluation/runs/<scenario>_cs*_seed*_*/` (not in git).
2. Aggregated to `results/scenario_evaluation/campaign_metrics.csv` and `results/campaign_size/{strong,weak}_summary.csv`.
3. `generate_tables` (missing base + recovered balanced wrapper) emitted `tables/table_P7_strong_campaign_results.csv` and `table_P8_weak_campaign_results.csv` (means over seeds).
4. Consolidation copied those CSVs into `experimental-2026-06-23/01_primary_ocslab_balanced/tables/`.

Verified numerically: P7/P8 F1 columns match `BALANCED_PUBLICATION_SUMMARY.md` and frozen summaries.

---

## 5. Ambiguities / unrecovered pieces

1. **Fifteen missing packages** listed in `MISSING_MODULES.md` — especially `run_refinement_fcgnn`, `SharedFleetConfiguration`, `_run_single_test`, `PUBLICATION_SCENARIOS`, `extract_run_metrics`, and base `generate_tables`.
2. **No committed descriptors / IF models / `processed/window_features.csv`** for the balanced run.
3. **λ and weight_decay** not in freeze YAML; only peer code defaults (0.25 / 5e-4).
4. **PCA dim** not in freeze YAML; peer default 8.
5. **Which cohesion formula** (centroid vs pairwise) the publication refinement used.
6. **How `minimum_cross_vehicle_support` was enforced** (freeze key present; no recovered consumer).
7. **Campaign↔GT matching** for P7/P8 (Jaccard≥0.5 is ablation documentation only).
8. **γ / η symbol mapping** not found in authoritative balanced artifacts (only β→cohesion 0.5 is explicitly labeled elsewhere).
9. **Peer defaults diverge from freeze** (`k_same` 10 vs 2; DBSCAN 1.2/10 vs 0.5/2; cohesion 0.85 vs 0.5) — freeze wins for publication numbers, but application code is missing.
10. Raw OCSLab path in master YAML points at an author OneDrive directory (external).

---

## 6. Quick reference — publication freeze values

```
τ=0.95  k_same=2  k_cross=5  cosine  fleet:benign-train-9D
GraphSAGE 9→64→32  mean-SAGEConv  ReLU  epochs=30  lr=0.01
(peer) wd=5e-4  λ=0.25  structure objective
DBSCAN: StandardScaler→PCA(8)→euclidean  eps=0.5  min_samples=2
gate: min_vehicles=2  min_cross_support=1  cohesion=0.5  fragment_centroid=0.85
fleet: 20×10=200 nodes  seeds={11,23,37,41,53,67,71,83,97,101}  cs={2,5,10}
```
