# ETA historical gate audit — did P7/P8 enforce `|C_k| >= η`?

**Verdict: B. NO_EVIDENCE_OF_ETA_GATE**

There is **no evidence** that the historical balanced publication run that produced tables P7/P8 enforced an **independent** post-DBSCAN campaign-size gate of the form `|C_k| >= η`.

**Recommendation:** Do **not** introduce η into the aligned CTT pipeline merely to match the current manuscript formula. Align CTT to the recoverable P7/P8 freeze gate keys, not to an unrecovered manuscript symbol.

**Non-actions (this audit):** no code changes, no result changes, no CTT run, no η selection.

---

## Question under investigation

The current manuscript defines the campaign gate as:

```text
r_k >= γ
|C_k| >= η
c_k >= β
```

The publication recovery audit found no historical value for η. This document asks a sharper question:

> Did the historical P7/P8 **implementation** actually enforce **any** independent post-DBSCAN minimum campaign cluster-size condition equivalent to `|C_k| >= η`?

---

## Four quantities that must not be conflated

| # | Quantity | Historical P7/P8 status | Evidence |
|---|----------|-------------------------|----------|
| 1 | **DBSCAN `min_samples` = 2** | **VERIFIED_FROM_ARTIFACT** | Freeze YAML / `parameter_search.csv` / selection report. Controls DBSCAN density/core-point formation **during clustering**, not campaign acceptance after clustering. Varied in search as `{2,3,4}` **independently** of gate columns. |
| 2 | **`minimum_distinct_vehicles` = 2** | **VERIFIED_FROM_ARTIFACT** | Freeze campaign-rule key. Candidate for manuscript γ (`r_k`); greek assignment in balanced artifacts **UNRECOVERABLE**. |
| 3 | **`minimum_cross_vehicle_support` = 1** | **VERIFIED_FROM_ARTIFACT** (value); enforcement **UNRECOVERABLE** | Freeze campaign-rule key. Searched as `{1,2}`. Name indicates cross-vehicle support, **not** `|C_k|`. Must **not** be treated as η. |
| 4 | **Minimum campaign cluster size `|C_k| >= η`** | **NO EVIDENCE in P7/P8 freeze / search / reconstructed gate** | Absent from freeze YAML, master YAML, parameter-search columns, selection report, and from recovered gate reconstructions that cite the publication pair. |

Items 1–3 are **not** evidence for item 4.

---

## Authoritative P7/P8 identity

| Layer | Path |
|-------|------|
| Numbers | Strong F1 `0.533 / 0.733 / 1.000`, Weak `0.067 / 0.500 / 0.717` |
| Archive | `experimental-2026-06-23/01_primary_ocslab_balanced/` |
| Producer | `new_experiments/final_end_to_end_publication_run_balanced/` @ `61a82038` |
| Freeze | `final_shared_fleet_configuration.yaml` (hash family `72dbfc17…`) |
| Executor | missing `run_refinement_fcgnn` / `_run_single_test` (never committed) |

Sources searched for this audit: recovered freeze configs, `parameter_search.csv`, selection reports, balanced audit JSON, frozen P7/P8 CSVs (column presence only), `MISSING_MODULES.md`, recovered peer code, `model_diversity_final*` gate freeze (non-authoritative stand-in), reviewer-ablation gate that cites the publication pair, git history / `origin/cursor/campaign-clustering`, prior `outputs/publication_metric_recovery/CAMPAIGN_GATE_AUDIT.md`.

---

## What the P7/P8 freeze actually contains

`final_shared_fleet_configuration.yaml` campaign-related keys (complete list of gate-adjacent freeze fields):

```yaml
dbscan_eps: 0.5
dbscan_min_samples: 2
fragment_consolidation_enabled: true
fragment_centroid_threshold: 0.85
minimum_distinct_vehicles: 2
minimum_cross_vehicle_support: 1
minimum_campaign_cohesion: 0.5
```

**Absent:** `min_cluster_size`, `minimum_cluster_size`, `eta`, `|C_k|`, or any synonym for a post-clustering size threshold.

The same absence holds for:

- `parameter_search.csv` (14 candidates; searchable dimensions listed below)
- `validation_scenarios/selection_report.md`
- `01_primary_ocslab_balanced/audit/original_vs_balanced_split.md`
- `balanced_master_experiment.yaml` (no η / `|C_k|` gate key)

### Joint parameter search dimensions (affirmative negative evidence)

Search columns on the run that produced the freeze:

| Searched | Values observed |
|----------|-----------------|
| `similarity_threshold`, `max_same_vehicle_neighbors`, `max_cross_vehicle_neighbors` | graph |
| `dbscan_eps`, `dbscan_min_samples` | clustering; `min_samples ∈ {2,3,4}` |
| `fragment_consolidation_enabled`, `fragment_centroid_threshold` | fragment merge |
| `minimum_distinct_vehicles` | `{2}` |
| `minimum_cross_vehicle_support` | `{1,2}` |
| `minimum_campaign_cohesion` | `{0.5}` |

**Size-like columns in that CSV: none.**

If an independent `|C_k| >= η` gate had been part of the publication configuration space that was validated and frozen for P7/P8, it would be expected to appear as a search and/or freeze key alongside the other campaign-rule fields. It does not.

DBSCAN `min_samples` was varied **without** any paired `|C_k|` column — further evidence that the search treated density clustering and campaign rules as separate families, and that `min_samples` was **not** standing in for η in that configuration artifact.

---

## Historical campaign decision logic (as far as evidence permits)

### `minimum_distinct_vehicles = 2`

| Aspect | Finding |
|--------|---------|
| Meaning | Minimum number of distinct vehicles (`r_k`) represented in a cluster for campaign acceptance |
| Value | **2** — **VERIFIED_FROM_ARTIFACT** |
| Manuscript γ | Candidate only; symbol assignment **UNRECOVERABLE** in balanced artifacts |
| Application code | Consumer lived in missing `SharedFleetConfiguration` / `run_refinement_fcgnn` — **WIRING UNRECOVERED** |
| Closest recovered predicates | Peer / ablation gates require `n_vehicles >= min_vehicles` (typically 2) |

### `minimum_cross_vehicle_support = 1`

| Aspect | Finding |
|--------|---------|
| Meaning (from name) | A minimum cross-vehicle support requirement (edges / neighbors / support count) |
| Value | **1** — **VERIFIED_FROM_ARTIFACT**; search also tested `2` |
| Is it η? | **No.** Not a cluster cardinality `|C_k|` threshold |
| Enforcement | **UNRECOVERABLE** — freeze key present; no recovered consumer in balanced path |
| Ablation note | Reviewer-ablation `config.yaml` stores `min_cross_vehicle_support: 1`, but recovered `apply_campaign_gate` in that harness does **not** apply a separate `|C_k|` check and documents acceptance as vehicles + cohesion (+ fragment merge) |

### `minimum_campaign_cohesion = 0.5`

| Aspect | Finding |
|--------|---------|
| Meaning | Minimum behavioural cohesion `c_k` for accepting a cluster as a campaign |
| Value | **0.5** — **VERIFIED_FROM_ARTIFACT** |
| Manuscript β | Labeled β in later ablation provenance that cites the publication pair (0.5 cohesion / 0.85 fragment) — **VERIFIED_FROM_ARTIFACT** for the numeric pair; balanced freeze itself stores the key without printing β |
| Formula | Centroid vs pairwise cosine — which one `run_refinement_fcgnn` used is **UNRECOVERABLE** |

### `fragment_centroid_threshold = 0.85`

| Aspect | Finding |
|--------|---------|
| Meaning | Fragment consolidation: merge already-accepted campaigns whose embedding centroids have cosine ≥ threshold |
| Value | **0.85** — **VERIFIED_FROM_ARTIFACT** |
| Is it η? | **No.** Post-acceptance merge, not `|C_k|` |
| Distinct from β | Explicitly distinguished in ablation provenance |

---

## Closest recovered gate predicates (none are P7/P8 emitters)

These show what *related* code did. They are **not** substitutes for the missing P7/P8 consumer.

### 1. Reviewer ablation `apply_campaign_gate` (cites publication pair 0.5 / 0.85)

Documented acceptance:

```text
not noise
AND n_vehicles >= min_vehicles
AND mean intra-cluster cosine >= cohesion_threshold
THEN optional fragment merge at fragment_merge_threshold
```

**No `|C_k| >= η` term.** This is the reconstructed gate used in ablation work that intentionally cites the publication cohesion/fragment pair.

### 2. `summarize_clusters` (recovered `campaign_clustering.py`)

Suspicious campaign:

```text
not noise
AND n_vehicles >= min_vehicles
AND mean_sim >= similarity_threshold
```

`cluster_size < 2` appears only when choosing how to compute mean similarity (avoid pairwise on singletons). It is **not** an independent campaign-size η gate.

### 3. `model_diversity_final` / tuned campaign gate (non-authoritative)

`CampaignGateConfig` / `final_selected_campaign_gate.yaml` fields: vehicles, anomalous ratio, weak-only ratio, membership confidence, cohesion, cross-model edges, etc.

**No `min_cluster_size` / `|C_k|` field.**

### 4. IEEE peer `FinalGnnFleetConfig` / `campaign_detection_experiment` (**NOT P7/P8**)

These peers **do** implement `size >= cfg.min_cluster_size` with default **10**, alongside `min_vehicles` and cohesion — but with **different** DBSCAN defaults (`eps=1.2`, `min_samples=10`) and **no** presence of that size key in the balanced publication freeze.

`MISSING_MODULES.md` explicitly lists this peer as a **closest stand-in, not equivalent**. Attributing peer `min_cluster_size=10` to historical P7/P8 would invent a freeze parameter that the freeze does not contain.

`campaign_detection_experiment` also maps HDBSCAN `min_cluster_size=cfg.dbscan_min_samples` — a **clustering** API argument, not evidence of a post-DBSCAN η gate in P7/P8.

---

## Manuscript formula vs historical implementation

The manuscript’s `|C_k| >= η` is a **paper-level** definition. Recoverable P7/P8 **implementation artifacts** do not include a corresponding configuration key or search dimension.

| Source type | Supports independent η gate for P7/P8? |
|-------------|----------------------------------------|
| Manuscript formula | Claims a three-term gate; **not** recovered executable evidence for the balanced run |
| P7/P8 freeze + parameter search | **Negative evidence** — no `|C_k|` / η key in the frozen/searched campaign-rule space |
| Missing `run_refinement_fcgnn` | Could theoretically hardcode an undocumented size check — **speculation only**; no blob, log, or config supports it |
| Peer IEEE `min_cluster_size=10` | Different path; **must not** be attributed to P7/P8 |
| Ablation gate citing publication pair | Reconstructs vehicles + cohesion + fragment; **no** `|C_k|` |

Residual speculation that a missing module hard-coded a size check without freeze exposure is **not** sufficient to choose verdict **C (UNRESOLVED)**. Verdict C requires affirmative evidence that such a gate existed. Here the freeze/search artifacts affirmatively omit it from the publication configuration space.

---

## Why not A or C

### Not A. `VERIFIED_ETA_GATE`

No verified value and no verified P7/P8 source for an independent `|C_k| >= η` predicate.

### Not C. `UNRESOLVED`

Peer size gates and the manuscript formula suggest size thresholds existed in **other** or **aspirational** formulations, but they do **not** constitute evidence that the **historical P7/P8 freeze path** enforced one. The authoritative freeze/search record for that path contains no such parameter.

---

## Verdict and recommendation

### **B. NO_EVIDENCE_OF_ETA_GATE**

There is no evidence that P7/P8 used an independent campaign-size gate `|C_k| >= η`.

**Recommend:** do **not** introduce η into the aligned CTT pipeline merely to match the manuscript. For methodology alignment to historical P7/P8, the recoverable campaign decision surface is:

```text
DBSCAN: eps=0.5, min_samples=2          # clustering only
gate:   minimum_distinct_vehicles = 2
        minimum_cross_vehicle_support = 1   # value verified; enforcement unrecovered
        minimum_campaign_cohesion = 0.5
fragment: fragment_centroid_threshold = 0.85 (enabled)
```

Any later manuscript edit that drops or redefines η is editorial; this audit does not modify the manuscript.

---

## Related documents

- `outputs/publication_metric_recovery/CAMPAIGN_GATE_AUDIT.md` — freeze keys; η already marked UNRECOVERABLE as a *value*
- `outputs/publication_metric_recovery/DBSCAN_PROVENANCE_AUDIT.md` — DBSCAN 0.5/2
- `outputs/ctt_aligned_publication/ETA_DBSCAN_SEPARATION_AUDIT.md` — decoupling DBSCAN `min_samples` from η in CTT code (PR #24 correction)
- `recovered_publication_pipeline/AUTHORITATIVE_PIPELINE_PARAMETER_AUDIT.md`
- `recovered_publication_pipeline/MISSING_MODULES.md`
