# CAMPAIGN_GATE_AUDIT.md

## Gate keys used by the balanced publication freeze

From `final_shared_fleet_configuration.yaml` (also embedded in `01_primary_ocslab_balanced/audit/original_vs_balanced_split.md` and `parameter_search.csv` winning row):

| Freeze key | Value | Greek symbol in authoritative balanced artifacts? |
|------------|-------|-----------------------------------------------------|
| `minimum_distinct_vehicles` | **2** | **NOT FOUND** as γ |
| `minimum_cross_vehicle_support` | **1** | **NOT FOUND** as η |
| `minimum_campaign_cohesion` | **0.5** | Labeled **β** only in later ablation docs citing this freeze |
| `fragment_consolidation_enabled` | **true** | — |
| `fragment_centroid_threshold` | **0.85** | Distinct from β |

**Grade:** values `VERIFIED_FROM_PUBLICATION_ARTIFACT`.  
**Grade for γ/η symbol assignment:** `UNRECOVERABLE` in balanced artifacts (do not invent).

There is **no** freeze key named `minimum_cluster_size`, `eta`, or `|C_k|` minimum in:

- `final_shared_fleet_configuration.yaml`
- `parameter_search.csv` search columns
- `balanced_master_experiment.yaml`

---

## A. DBSCAN (separate from gate)

| Parameter | Publication freeze value | Source | Used by P7/P8? |
|-----------|--------------------------|--------|----------------|
| `dbscan_eps` | **0.5** | FREEZE YAML + selection_report + parameter_search winner | **Yes** — freeze selected for shared config of balanced run (`VERIFIED_FROM_PUBLICATION_ARTIFACT`) |
| `dbscan_min_samples` | **2** | same | **Yes** (same grade) |
| metric | euclidean (on PCA space) | peer `run_dbscan` CODE | Peer path; wiring via missing SharedFleet consumer (`WIRING UNRECOVERED`) |
| preprocessing | StandardScaler → PCA(8) | peer `DbscanProjector` CODE | same |

**Do not equate `dbscan_min_samples` with campaign-gate η.**  
No publication artifact maps them. Parameter search varies `dbscan_min_samples` (2/3/4) **independently** of campaign-rule columns (`minimum_distinct_vehicles`, `minimum_cross_vehicle_support`, `minimum_campaign_cohesion`).

---

## B. Campaign gate (γ, η, β)

### Recovered freeze values (authoritative for the shared config that scored P7/P8)

| Role (user terminology) | Freeze field | Value | Evidence connected to P7/P8 |
|-------------------------|--------------|-------|-----------------------------|
| Distinct-vehicle support | `minimum_distinct_vehicles` | **2** | Freeze + search winner written before test loop in runner | 
| Cross-vehicle support | `minimum_cross_vehicle_support` | **1** | same |
| Behavioral cohesion | `minimum_campaign_cohesion` | **0.5** | same |
| Minimum cluster size η | — | **UNRECOVERABLE** | Not in freeze/search; peer `FinalGnnFleetConfig.min_cluster_size=10` is **DEFAULT ONLY** and **not** present in publication freeze |

**Function/caller that applied the gate to clusters:** missing `SharedFleetConfiguration` consumer inside missing `run_refinement_fcgnn` / `_run_single_test`.  
**Grade:** freeze *values* `VERIFIED_FROM_PUBLICATION_ARTIFACT`; *application code* `UNRECOVERABLE`.

### Peer cohesion formulas (which one was used: UNRECOVERABLE)

1. Centroid cosine — `compute_cluster_behavioral_cohesion` in recovered `final_gnn_fleet_decision_experiment.py`
2. Mean pairwise cosine — `mean_intra_cluster_similarity` in `campaign_clustering.py`

IEEE peer also defaults `min_behavioral_cohesion=0.85` and `min_cluster_size=10` — **DEFAULT ONLY**, overridden for publication by freeze cohesion **0.5** (consumer missing).

---

## C. Membership metrics (evaluation, not gate)

From saved columns (`VERIFIED_FROM_PUBLICATION_ARTIFACT`):

**Level: vehicle** (not descriptor/node).

Let:

- `A_ok` = `attacked_vehicles_correctly_included`
- `A_miss` = `attacked_vehicles_missed`
- `B_in` = `benign_vehicles_included`
- `S_pred` = `predicted_campaign_size`
- `S_true` = `true_campaign_size`

Identities:

```
S_pred = A_ok + B_in
S_true = A_ok + A_miss
membership_recall    = A_ok / S_true
membership_precision = A_ok / S_pred   (0 when S_pred=0)
membership_purity    = membership_precision
membership_f1        = 2PR/(P+R)
completeness         = campaign_recall   # NOT membership_recall
```

**Benign vehicles inside a predicted campaign are membership false positives** (they inflate `S_pred` without increasing `A_ok`).

Exact membership vehicle-assignment code (which vehicles belong to the predicted campaign set): **`UNRECOVERABLE`** (emitter missing). Artifact identities constrain the *aggregates* only.

---

## D. Fragmentation

From saved strong/weak rows (`VERIFIED_FROM_PUBLICATION_ARTIFACT`):

```
fragments_per_true_campaign = n_predicted_campaign_clusters
fragmentation_rate = 1{ n_predicted_campaign_clusters > n_true_campaign_clusters }
```

Interpretation consistent with artifacts (with `n_true=1`):

- If no predicted campaign: `frag_per=0`, `frag_rate=0` (miss is not a “fragment”)
- If exactly one predicted campaign: `frag_per=1`, `frag_rate=0`
- If two+ predicted campaigns: `frag_per=n_pred`, `frag_rate=1`

DBSCAN noise is not a fragment under these identities (noise is outside `n_predicted_campaign_clusters`).

Aggregation to P7/P8: mean over seeds within `campaign_size` (`VERIFIED_FROM_PUBLICATION_ARTIFACT`).

---

## E. Incorrect merging

| Item | Finding | Grade |
|------|---------|-------|
| Where it matters | Unrelated Multi-Vehicle Incidents; mean rate **0.400** = P6 | Artifact `VERIFIED` |
| Per-seed values | Binary `{0,1}`; 4/10 seeds = 1 | Artifact `VERIFIED` |
| When `incorrect_merging=1` | Always `n_predicted_campaign_clusters=1` and `unrelated_clusters_merged=1` | Artifact `VERIFIED` |
| Denominator for *rate* | Number of unrelated scenario seeds (10); rate = mean of binary flag | Artifact `VERIFIED` |
| Semantic (peer) | `n_gt > 1` and `n_accepted == 1` (unrelated incidents merged into one campaign) in `false_campaign_metrics.py` | Peer `INFERRED` for P7/P8 |
| Exact predicate in `extract_run_metrics` / `safety_row` | Missing | `UNRECOVERABLE` |

Not measuring: benign∪malicious membership contamination (that is separate `benign_vehicles_included`), nor strong/weak campaign F1.
