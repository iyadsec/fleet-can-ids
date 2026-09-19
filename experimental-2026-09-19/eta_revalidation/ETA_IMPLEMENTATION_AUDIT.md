# ETA implementation audit — prospective paper methodology

**Experiment root:** `experimental-2026-09-19/eta_revalidation/`  
**Historical reference (untouched):** `experimental-2026-06-23/01_primary_ocslab_balanced/`  
**Primary verdict:** `STOP_NO_VALIDATION_SPLIT`  
**Concurrent blocker:** `STOP_METRIC_IMPLEMENTATION_UNRECOVERED`

---

## Methodology retained (paper)

Independent campaign gate after DBSCAN:

```text
r_k  >= γ     # distinct vehicles in C_k
|C_k| >= η    # anomaly descriptors / nodes in C_k
c_k  >= β     # centroid behavioural cohesion
```

Execution order (implemented in `src/evaluation/publication_fleet_core.py`):

```text
graph → GraphSAGE → StandardScaler→PCA(8) → DBSCAN → candidate C_k → campaign gate
```

η is **`min_campaign_cluster_size`**: required explicitly, default `None`, never derived from `dbscan_min_samples`.

---

## Recovered frozen configuration (non-η)

| Component | Value | Evidence |
|-----------|-------|----------|
| Cosine τ | 0.95 | FREEZE |
| k_same / k_cross | 2 / 5 | FREEZE |
| GNN input | 9-D recovered order | CODE peer |
| Architecture | 9→64→32, mean SAGEConv, ReLU after conv1 | FREEZE + CODE |
| Edge weights in SAGEConv | not consumed | CODE |
| Epochs / Adam lr / wd / λ | 30 / 0.01 / 5e-4 / 0.25 | FREEZE + peer CODE |
| Supervision | structure (link + anomaly-score) | CODE |
| DBSCAN | Scaler→PCA(8)→euclidean, eps=0.5, min_samples=2 | FREEZE + CODE |
| γ candidate | `minimum_distinct_vehicles=2` | FREEZE |
| β candidate | `minimum_campaign_cohesion=0.5` | FREEZE |
| Cross-vehicle support | `minimum_cross_vehicle_support=1` | FREEZE (enforcement UNRECOVERABLE) |
| Fragment | enabled, centroid threshold 0.85 | FREEZE |

**Not introduced from IEEE/legacy peers:** `min_cluster_size=10`, DBSCAN 1.2/10, attack-type gate rules.

**Historical η:** no value claimed; prior audit verdict was `NO_EVIDENCE_OF_ETA_GATE` for P7/P8 freeze enforcement. This experiment treats η as a **prospective** paper parameter.

---

## η candidate set (recorded before any evaluation)

```text
eta ∈ {2, 3, 5, 10}
```

Recorded in `eta_candidate_set_predeclared.json` **before** attempting validation evaluation. Not expanded after the STOP.

---

## Implementation status

| Item | Status |
|------|--------|
| Independent `|C_k| >= η` gate | **Implemented** (`PublicationFleetConfig.min_campaign_cluster_size`) |
| DBSCAN `min_samples` fixed at 2 unless explicitly overridden | **Yes** |
| Unit tests for independence / stage order / \|C_k\| vs r_k / no label leakage | **Yes** (`tests/test_ctt_aligned_publication_fleet.py`) |
| Validation-only η selection | **NOT RUN** — no runnable fleet validation split |
| Final OCSLab re-run | **NOT RUN** |
| CTT | **NOT RUN** (forbidden) |
| Historical P7/P8 overwrite | **None** |

---

## Why STOP (validation)

See `FINAL_VERDICT.md`. Summary:

- Historical `validation_manifest.csv` lists V0–V4 validation scenarios used by missing `joint_parameter_search` / `build_mixed_validation_suite`.
- Manifest is **metadata only**: most `event_ids` empty; no scenario descriptor CSVs; no balanced `window_features.csv` / IF models / descriptors in git or this environment.
- OCSLab raw traces are not present under `Dataset/` here.
- Substituting the final test partition is forbidden.
- Reconstructing synthetic validation fleets would invent data, not recover the primary methodology.

## Concurrent metric blocker

Even if validation descriptors were restored, Campaign F1 / Membership F1 / fragmentation emitters require the unrecovered P7/P8 matcher (`extract_run_metrics`). Ablation Jaccard≥0.5 is **not** authorised as the publication matcher. Inventing a replacement metric is forbidden for this experiment.

---

## Outputs under this directory

| Path | Content |
|------|---------|
| `ETA_IMPLEMENTATION_AUDIT.md` | this file |
| `eta_candidate_set_predeclared.json` | candidate η set before evaluation |
| `frozen_fleet_configuration.json` | recovered non-η freeze (+ η unset) |
| `eta_selection.json` | selection **not** performed |
| `ETA_VALIDATION_RESULTS.csv` | empty header + STOP note row |
| `ETA_CLUSTER_SIZE_ANALYSIS.csv` | empty header + STOP note row |
| `final/` | placeholders only — no final run |
| `HISTORICAL_VS_ETA_REVALIDATED.md` | comparison deferred |
| `ETA_DECISION_IMPACT.md` | impact deferred |
| `REPRODUCIBILITY_MANIFEST.json` | reproducibility record |
| `FINAL_VERDICT.md` | `STOP_NO_VALIDATION_SPLIT` |
