# VALIDATION_DESCRIPTOR_AUDIT.md

## Status: **GENERATED** (validation split only)

Source: reconstructed scored windows with `split == validation` (20,669 windows).  
Output: `artifacts/validation_descriptors.csv` (local artifact; large file gitignored).

---

## dᵢ vs gᵢ

| Symbol | Meaning | Columns | Verified in |
|--------|---------|---------|-------------|
| **dᵢ** | 24-D behavioural window descriptor | `BEHAVIOURAL_FEATURE_COLUMNS` | `src/features/feature_extractor.py` |
| **gᵢ** | 9-D GraphSAGE **input** representation | `GNN_FEATURE_COLUMNS` | `src/evaluation/publication_fleet_core.py` |

### dᵢ (24-D) — exact order

`frame_count`, `unique_can_id_count`, `can_id_entropy`, `most_common_can_id_ratio`, `mean_inter_arrival_time`, `std_inter_arrival_time`, `mean_dlc`, `std_dlc`, `byte_mean_0`…`byte_mean_7`, `byte_std_0`…`byte_std_7`

### gᵢ (9-D) — exact order (authoritative)

1. `anomaly_score`
2. `message_rate` (= `frame_count` in behaviour view)
3. `frame_count`
4. `burstiness` (= `std_inter_arrival_time / (|mean_inter_arrival_time| + ε)`)
5. `mean_inter_arrival_time`
6. `std_inter_arrival_time`
7. `can_id_entropy`
8. `most_common_can_id_ratio`
9. `payload_entropy` (entropy of abs(`byte_mean_*`) row-normalised)

Derived via `build_behavior_view_descriptors` + `compute_payload_entropy`. **No GraphSAGE training** was run; gᵢ columns are prepared inputs only.

---

## Provenance columns retained

| Field | Present |
|-------|---------|
| descriptor ID | `descriptor_id` |
| vehicle | `vehicle_model` |
| source trace | `source_file` / `relative_source` / `segment_id` |
| window | `window_id`, `start_frame_idx`, `end_frame_idx` |
| split | `validation` only |
| label | evaluation-only (`label`, `attack_type`) |
| anomaly score | `anomaly_score` (+ vehicle FPR≤5% `threshold`, `predicted_label`) |

### Isolation from test

- Descriptors exported **only** from validation windows.
- Test windows were scored for vehicle-level audit but **not** written into the validation descriptor table or scenario builder inputs.

### Counts

| Split | Windows / descriptors |
|-------|------------------------|
| validation descriptors | **20,669** |
| test descriptors exposed to scenario builder | **0** |
