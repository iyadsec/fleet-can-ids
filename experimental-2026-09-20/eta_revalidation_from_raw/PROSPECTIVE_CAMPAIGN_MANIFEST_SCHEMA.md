# PROSPECTIVE_CAMPAIGN_MANIFEST_SCHEMA.md

**Status:** NEW / PROSPECTIVE — design only  
**Companion:** `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1.md`  
**Purpose:** Make controlled descriptor-level campaign construction completely auditable.

This schema is **not** a claim that historical P7/P8 manifests are recoverable. It defines the audit trail for the **new prospective** protocol.

---

## 1. Units of record

Emit **one row per descriptor node** placed into a constructed fleet scenario (maximum 200 nodes per scenario instance under the frozen budget).

Optional companion tables (same run id):

| Table | Grain | Required? |
|-------|-------|-----------|
| `campaign_descriptor_manifest` | one row / descriptor | **Yes** |
| `campaign_run_header` | one row / (scenario, seed, campaign_size, \(s\)) | **Yes** |
| `prototype_registry` | one row / prototype vector used | **Yes** when any transform applied |
| `original_descriptor_store` | content-addressed original 24-D vectors | **Yes** (or equivalent side store) |

---

## 2. Required columns — `campaign_descriptor_manifest`

| Column | Type | Description |
|--------|------|-------------|
| `protocol_id` | string | Constant: `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1` |
| `protocol_status` | string | Constant: `NEW_PROSPECTIVE` |
| `run_id` | string | Unique id for one constructed scenario instance |
| `scenario` | string | `S0` / `S1` / `S2` / `S3` / `S4` |
| `seed` | int | Construction RNG seed |
| `campaign_size` | int | Attacked-vehicle count (`0` for S0; `1` for S1; configured size otherwise) |
| `fleet_size` | int | Must be `20` under frozen budget |
| `descriptors_per_vehicle` | int | Must be `10` under frozen budget |
| `node_index` | int | Position in fleet graph node list `[0, 199]` |
| `vehicle_token` | string | Scenario vehicle instance id |
| `source_vehicle` | string | Original OCSLab / catalog vehicle identity |
| `source_trace` | string | Source trace / file name |
| `window_index` | int | Window index within the source trace |
| `event_id` | string | Stable event / descriptor id if present |
| `attack_type` | string | Attack family/type of the **source window** |
| `original_anomaly_score` | float | Score **before** any coordination transform (unchanged by blend) |
| `score_band` | string | `none` / `weak` / `strong` / `benign` per frozen thresholds |
| `original_descriptor_hash` | string | Hash of the original 24-D behavioural vector (see §4) |
| `coordination_strength` | float | \(s\) applied to this row (`0.0` if untouched) |
| `coordination_applied` | bool | Whether `apply_coordination_strength` rewrote this row’s \(C\) |
| `prototype_id` | string | Id into `prototype_registry` (empty if none) |
| `prototype_hash` | string | Hash of \(\tilde{p}\) **after** clip-to-target-bounds used in the blend, or empty |
| `transformed_descriptor_hash` | string | Hash of post-transform 24-D behavioural vector (equals original hash if untouched) |
| `GT_campaign_id` | string | Shared campaign id for coordinated members; distinct incident id for S2; empty for non-members |
| `ground_truth_malicious` | int | `0` / `1` |
| `ground_truth_campaign_member` | int | `0` / `1` |
| `scenario_role` | string | e.g. `benign` / `isolated` / `unrelated` / `coordinated` |
| `transform_method` | string | `none` or `prototype_blend_with_bounded_noise` |
| `noise_sigma` | float | \(0.02(1-s)\) when applied; `0` otherwise |
| `split` | string | Descriptor split provenance (`validation` / `test` / …) — must not mix forbidden splits |

---

## 3. Required columns — `campaign_run_header`

| Column | Type | Description |
|--------|------|-------------|
| `protocol_id` | string | `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1` |
| `run_id` | string | Matches descriptor rows |
| `scenario` | string | S0–S4 |
| `seed` | int | |
| `campaign_size` | int | |
| `coordination_strength` | float | Configured \(s\) for the run (`0` for S0/S1/S2 under V1) |
| `s_selection_status` | string | `CANDIDATE_UNEVALUATED` / `FROZEN_PROSPECTIVE` (never `HISTORICAL_RECOVERED`) |
| `eta_status` | string | Constant until later: `UNSELECTED_CANDIDATES_{2,3,5,10}` |
| `metric_protocol` | string | Intended later gate: `revised_jaccard_0.5_v3` (not executed here) |
| `fleet_size` | int | `20` |
| `total_nodes` | int | `200` when budget satisfied |
| `budget_ok` | bool | |
| `prototype_id` | string | Shared prototype for S3/S4; empty for controls |
| `attack_family_default` | string | e.g. `malfunction` |
| `strong_threshold` | float | `0.80` |
| `weak_threshold` | float | `0.55` |
| `construction_label` | string | `controlled_descriptor_level_campaign_construction` |
| `historical_equivalence` | string | Constant: `NOT_EQUIVALENT_TO_HISTORICAL_P7_P8` |

---

## 4. Hashing rules

For audit reproducibility:

1. Serialize the 24 behavioural features in the exact `BEHAVIOURAL_FEATURE_COLUMNS` order.  
2. Use IEEE float64 little-endian binary of each finite value; encode non-finite as a documented sentinel.  
3. Compute `sha256` hex digest → `original_descriptor_hash` / `transformed_descriptor_hash`.  
4. Prototype hash: same serialization over the 24-D \(\tilde{p}\) actually used **after** target-subset clipping.  
5. Store full original vectors in `original_descriptor_store` keyed by `original_descriptor_hash` so audits need not re-extract CAN to verify the pre-transform state.

Do **not** hash post-GNN embeddings in place of the behavioural descriptor hashes.

---

## 5. `prototype_registry` columns

| Column | Type | Description |
|--------|------|-------------|
| `prototype_id` | string | Stable id |
| `prototype_hash` | string | Pre-clip mean vector hash (optional) and/or post-clip hash |
| `attack_type` | string | Family used in `compute_campaign_prototype` |
| `pool_description` | string | Explicit pool (e.g. `validation_split_all_matching_attack_type`) |
| `n_source_rows` | int | Rows averaged |
| `feature_columns_json` | string | Exact \(C\) list |
| `classification` | string | `RECOVERED_METHOD` + note that this instance is **prospective** |

---

## 6. Forbidden omissions

A campaign artifact set is incomplete if any coordinated (transformed) row lacks:

- original score + original descriptor hash  
- \(s\), prototype id/hash  
- transformed descriptor hash  
- `GT_campaign_id`

S2 rows must show **no shared** prototype across independent incidents.

---

## 7. Non-execution note

This schema is a **design**. Generating manifests, hashing descriptors, or building campaigns is **out of scope** for the design-only task.
