# DBSCAN_PROVENANCE_AUDIT.md

## Why both `0.5/2` and `1.2/10` exist

They are **different layers**, not two competing publication winners.

| Configuration | Where it lives | Experiment / role | Connected to P7/P8? |
|---------------|----------------|-------------------|---------------------|
| **eps=0.5, min_samples=2** | `final_shared_fleet_configuration.yaml`; `parameter_search.csv` winner; `selection_report.md`; `01_primary_ocslab_balanced/audit/original_vs_balanced_split.md` | Balanced end-to-end publication shared fleet config (selected on validation, then applied to test runs) | **Yes** — freeze of the run that produced `campaign_metrics` / P7/P8 | 
| **eps=1.2, min_samples=10** | Defaults in recovered peer source: `FinalGnnFleetConfig`, `CampaignDetectionConfig`, `run_dbscan` call defaults / `campaign_clustering.run_campaign_clustering` signature | IEEE / campaign-detection peer modules **not** overridden unless caller passes freeze | **No evidence** it generated P7/P8 |

**Grade:** freeze `0.5/2` → `VERIFIED_FROM_PUBLICATION_ARTIFACT` for P7/P8 config.  
**Grade:** code defaults `1.2/10` → `VERIFIED_FROM_SOURCE` as defaults only (`DEFAULT ONLY` for publication).

---

## Detailed provenance

### A. Publication freeze `0.5 / 2`

| Item | Detail |
|------|--------|
| VALUE | `dbscan_eps=0.5`, `dbscan_min_samples=2` |
| SOURCE FILE | `…/configs/final_shared_fleet_configuration.yaml` |
| FUNCTION/CALLER | Written by missing `joint_parameter_search` → `SharedFleetConfiguration`; consumed by missing `_run_single_test` / `run_refinement_fcgnn` |
| EVIDENCE USED BY P7/P8 | Freeze hash tied to master `72dbfc17…`; runner writes this YAML before enumerating test runs; audit JSON in `original_vs_balanced_split.md` records same values for balanced run |
| Search stage | `parameter_search.csv` shows winner under stages `dbscan` then `campaign_rules` with these values |
| Other candidates tested | eps∈{0.5,0.8}, min_samples∈{2,3,4} — **never 1.2/10** in this search grid |

Preprocessing for DBSCAN in peer code used by GNN fleet path: StandardScaler → PCA(n_components=8) → euclidean DBSCAN (`campaign_clustering.DbscanProjector` / `run_dbscan`).  
**Wiring that applied freeze eps/min_samples into that peer:** `UNRECOVERABLE` (missing SharedFleet consumer), but freeze *values* are authoritative.

### B. Peer defaults `1.2 / 10`

| Item | Detail |
|------|--------|
| VALUE | `dbscan_eps=1.2`, `dbscan_min_samples=10` |
| SOURCE FILE | `recovered_publication_pipeline/src/evaluation/final_gnn_fleet_decision_experiment.py` (`FinalGnnFleetConfig`); `campaign_detection_experiment.py`; `campaign_clustering.py` helper defaults |
| FUNCTION | `cluster_gnn_embeddings` / `run_dbscan` when cfg left at defaults |
| EVIDENCE FOR P7/P8 | **None.** Search grid + freeze disagree. Label: **DEFAULT ONLY** |
| Publication tables generated with these defaults alone? | **No evidence** |

### C. HDBSCAN note (not P7/P8)

`campaign_detection_experiment.py` maps `min_cluster_size=cfg.dbscan_min_samples` for **HDBSCAN**, not for the publication DBSCAN gate. That must **not** be read as η for P7/P8.

---

## Separation checklist (required)

| Concern | Publication value | Must not confuse with |
|---------|-------------------|------------------------|
| DBSCAN `min_samples` | **2** (freeze) | Campaign-gate η / `|C_k|` minimum |
| DBSCAN `eps` | **0.5** (freeze) | Cohesion β=0.5 (different quantity) |
| Campaign `minimum_distinct_vehicles` | **2** (freeze) | DBSCAN `min_samples` (same number, different role) |
| Peer `min_cluster_size` | **10** default | Publication freeze (absent) |

---

## Bottom line

- **Winner for P7/P8 config:** `eps=0.5`, `min_samples=2` from the shared fleet freeze.  
- **`1.2/10`:** leftover peer defaults for modules that were not the freeze consumer.  
- **η as minimum campaign cluster size:** not recovered from freeze; **do not set η := min_samples**.
