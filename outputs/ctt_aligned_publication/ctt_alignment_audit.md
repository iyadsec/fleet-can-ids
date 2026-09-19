# CTT alignment audit — primary OCSLab vs previous CTT fleet

**Authoritative primary artifacts:** `experimental-2026-06-23/01_primary_ocslab_balanced/`  
**Master config hash:** `72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e`  
**Freeze YAML:** `recovered_publication_pipeline/new_experiments/final_end_to_end_publication_run_balanced/configs/final_shared_fleet_configuration.yaml`  
**Shared implementation:** `src/evaluation/publication_fleet_core.py`

Publication headline numbers (unchanged; not re-run): Strong Campaign F1 `0.533 / 0.733 / 1.000`, Weak `0.067 / 0.500 / 0.717`, unrelated merge `0.400`.

---

## Audit table

| Component | Authoritative OCSLab implementation | Previous CTT implementation | Difference | Required correction |
|-----------|-------------------------------------|-----------------------------|------------|---------------------|
| Descriptor store | 24-D `BEHAVIOURAL_FEATURE_COLUMNS` | 32-D CTT `LOCAL_FEATURE_COLUMNS` (JSON vector) | Schema differs (dataset-specific) | Keep CTT local schema; derive same 9-D GNN view |
| GraphSAGE input `g_i` | Exact 9-D `GNN_FEATURE_COLUMNS` + benign scaler | Full 32-D descriptor vector | Wrong input dim/features | Use shared `prepare_gnn_fleet_node_matrix` |
| `message_rate` | Set to `frame_count` in behavior view | Real messages/sec in CTT features | Derivation differed | Use publication `build_behavior_view_descriptors` |
| `burstiness` | `std_IAT / (|mean_IAT|+1e-9)` | Absent as named column | Missing | Derive identically |
| `payload_entropy` | Entropy of abs(`byte_mean_*`) | Absent | Missing | Derive via `compute_payload_entropy` |
| Graph construction | Cosine constrained kNN; τ=**0.95**, k_same=**2**, k_cross=**5** | τ=0.85, single knn_cap=10, cross_cap=5 | τ and k_same wrong | Freeze values |
| Cosine calculation | sklearn cosine distance → `1-d` after feature prep | Same idea on StandardScaler of full vector | Feature space differed | Cosine on 9-D scaled GNN matrix |
| Edge weights in SAGEConv | Stored as `edge_attr`; **not** passed to SAGEConv | No edge_attr; topology only | Compatible intent | Keep: store weights, no SAGEConv edge_attr |
| GraphSAGE architecture | 9→64→32, mean aggr, ReLU after conv1 | 32→64→32, mean, ReLU | Input dim | Shared `GraphSAGEFleetCorrelator` |
| Training objective | Structure: `L_link + λ L_score`, λ=0.25 | Node-feature MSE reconstruction | Different loss | Shared `train_graphsage_fleet_correlation` |
| Optimizer | Adam lr=**0.01**, wd=**5e-4**, 30 epochs | Adam lr=1e-3, no wd, 50 epochs default in helper | Wrong hypers | Freeze + peer defaults |
| Embedding preprocess | StandardScaler → PCA(8) | None (raw embeddings) | Missing | Shared `run_dbscan` |
| DBSCAN metric | **euclidean** (PCA space) | **cosine** | Wrong | Euclidean |
| DBSCAN eps / min_samples | **0.5 / 2** (freeze) | 0.8 / 2 | eps wrong | Freeze |
| Campaign gate | r_k≥2, \|C_k\|≥η, c_k≥0.5 centroid cohesion | n_vehicles≥2 × attack-family cohesion | Used attack_type | Shared centroid gate; no attack_type |
| Cohesion formula | Centroid cosine (IEEE peer); pairwise also exists | Attack-family count inverse | Wrong | `compute_cluster_behavioral_cohesion` |
| γ / η / β | Freeze: min_vehicles=2, cohesion=0.5; η **UNRECOVERABLE** | N/A | Greek η unmapped; PR #24 wrongly used dbscan_min_samples | Decouple: η=`min_campaign_cluster_size` required explicitly; γ/β candidate freeze keys only |
| Fragment merge | freeze threshold 0.85 | None | Missing | Embedding-centroid merge ≥0.85 |
| Matching (P7/P8) | **UNRECOVERED** (`extract_run_metrics` missing) | Best cluster vehicle overlap | Unknown vs publication | Ablation peer Jaccard≥0.5 + STOP note |
| Seeds | 11,23,37,41,53,67,71,83,97,101 | Same list | None | Preserve |
| Node budget | 20×10=200 (OCSLab) | CTT scenario sampling (dataset-driven) | Unavoidable | Document; do not fabricate |

---

## Files / functions changed

| Path | Change |
|------|--------|
| `src/evaluation/publication_fleet_core.py` | **New** shared publication fleet core |
| `src/experiments/{data_splits,vehicle_identity,local_descriptor_normalisation}.py` | **New** minimal shared scaler/identity helpers |
| `src/ctt/fleet_campaign.py` | Replaced MSE/cosine-DBSCAN/attack-type gate with shared path |
| `src/ctt/fleet_graph.py` | Replaced ad-hoc graph with publication constrained-kNN adapter |
| `src/ctt/scenarios.py` | Separated A/B/C; calls shared inference + post-hoc eval |
| `src/ctt/constants.py` | Graph params → freeze τ=0.95, k_same=2, k_cross=5 |
| `src/models/__init__.py` | Lazy imports (break circular import for GraphSAGE) |
| `scripts/run_ctt_aligned_publication.py` | **New** aligned experiment runner |
| `tests/test_ctt_aligned_publication_fleet.py` | **New** unit/smoke tests |

**Not modified:** OCSLab publication results under `experimental-2026-06-23/01_primary_ocslab_balanced/`, manuscript/LaTeX.

---

## Authoritative source of every frozen parameter

| Parameter | Value | Source grade |
|-----------|-------|--------------|
| τ | 0.95 | FREEZE YAML |
| k_same | 2 | FREEZE YAML |
| k_cross | 5 | FREEZE YAML |
| DBSCAN eps / min_samples | 0.5 / 2 | FREEZE YAML |
| min_vehicles (γ candidate) | 2 | FREEZE `minimum_distinct_vehicles` (symbol γ UNRECOVERABLE) |
| cohesion β candidate | 0.5 | FREEZE `minimum_campaign_cohesion` (β labeled in ablation docs) |
| fragment threshold | 0.85 | FREEZE (NOT η) |
| cross_vehicle_support | 1 | FREEZE (NOT η; enforcement UNRECOVERABLE) |
| epochs / lr | 30 / 0.01 | FREEZE `graphsage.*` + CODE |
| wd / λ | 5e-4 / 0.25 | Peer CODE only (not in freeze YAML) |
| PCA dims | 8 | Peer CODE only |
| 9-D feature list | see GNN_FEATURE_COLUMNS | Peer CODE `final_gnn_fleet_decision_experiment` |
| η (\|C_k\|) | **UNRECOVERABLE** | Must be supplied as `min_campaign_cluster_size`; never from `dbscan_min_samples` |

---

## Remaining ambiguities (do not invent further)

1. Exact P7/P8 campaign↔GT matcher (`run_refinement_fcgnn` / `extract_run_metrics` missing).
2. Whether publication refinement used centroid vs pairwise cohesion (gate here uses **centroid**, matching the user-specified formula and IEEE peer).
3. How `minimum_cross_vehicle_support=1` was enforced beyond `r_k≥2`.
4. Historical η (`\|C_k\|`) — **UNRECOVERABLE**; see `ETA_DBSCAN_SEPARATION_AUDIT.md`. Prospective CTT requires explicit `--eta`.
5. Real can-train-and-test dataset absent in this environment — prior numerical outputs may be from a **synthetic fixture** and/or the incorrect η:=2 coupling; do not treat them as historically justified.
