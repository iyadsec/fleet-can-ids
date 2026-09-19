# ETA / DBSCAN separation audit (PR #24 correction after PR #25)

**Verdict companion:** `READY_FOR_ETA_DECISION`  
**Scope:** Decouple FLEET-GUARD campaign-size threshold η from DBSCAN `min_samples`.  
**Non-goals:** No CTT re-run; no historical OCSLab / P7 / P8 edits; no invented historical η; no parameter tuning.

---

## Core statement

> **DBSCAN `min_samples` and the FLEET-GUARD campaign-size threshold η are distinct parameters operating at different stages of the pipeline.**

| Stage | Parameter | Role |
|-------|-----------|------|
| Clustering | `dbscan_eps`, `dbscan_min_samples` | Density / core-point formation inside DBSCAN |
| Post-clustering campaign gate | `min_campaign_cluster_size` (η), `minimum_distinct_vehicles`, `minimum_campaign_cohesion` | Accept/reject formed clusters as campaigns via `\|C_k\| >= η` (and vehicle / cohesion checks) |

They may share a numeric value in some configuration, but there is **no evidence** that the historical P7/P8 publication run explicitly coupled them.

---

## Verified DBSCAN (unchanged)

| Parameter | Value | Evidence status |
|-----------|-------|-----------------|
| `dbscan_eps` | **0.5** | **VERIFIED_FROM_ARTIFACT** — `final_shared_fleet_configuration.yaml` |
| `dbscan_min_samples` | **2** | **VERIFIED_FROM_ARTIFACT** — same freeze YAML |

These remain the recovered P7/P8 publication DBSCAN settings. This correction does **not** change them.

---

## Campaign-gate parameter audit (freeze keys vs paper symbols)

Freeze source:  
`recovered_publication_pipeline/new_experiments/final_end_to_end_publication_run_balanced/configs/final_shared_fleet_configuration.yaml`

| Paper symbol | Code / config parameter | Value | Semantic meaning | Evidence status |
|--------------|-------------------------|-------|------------------|-----------------|
| **γ** (candidate) | `minimum_distinct_vehicles` | **2** | Minimum number of distinct vehicles represented in a cluster (`r_k` / vehicle support) for campaign acceptance | Value: **VERIFIED_FROM_ARTIFACT** (freeze). Symbol γ assignment to this key: **UNRECOVERABLE** in balanced publication artifacts (see PR #25 / `AUTHORITATIVE_PIPELINE_PARAMETER_AUDIT.md`) |
| **η** | `min_campaign_cluster_size` | **UNRECOVERABLE HISTORICAL VALUE** | Post-clustering minimum campaign cluster size `\|C_k\|` | **UNRECOVERABLE** — freeze YAML has **no** `\|C_k\|` / `min_cluster_size` / η key. Must be supplied explicitly for prospective CTT |
| **β** (candidate) | `minimum_campaign_cohesion` | **0.5** | Minimum behavioural cohesion `c_k` for campaign acceptance | Value: **VERIFIED_FROM_ARTIFACT** (freeze). Labeled β in later ablation docs that cite the publication pair: **VERIFIED_FROM_ARTIFACT** (ablation provenance), but balanced runner itself does not print the greek letter |
| *(not η)* | `minimum_cross_vehicle_support` | **1** | Freeze key present beside vehicle / cohesion gates. Exact consumer in missing `run_refinement_fcgnn` / SharedFleetConfiguration is unrecovered. **Not** reinterpreted as η | Value: **VERIFIED_FROM_ARTIFACT**. Meaning / enforcement: **UNRECOVERABLE** beyond the named freeze key. **Not** used as `\|C_k\|` |
| *(not η)* | `fragment_centroid_threshold` | **0.85** | Fragment consolidation: merge qualifying campaigns whose embedding centroids have cosine ≥ threshold | Value: **VERIFIED_FROM_ARTIFACT**. **Not** η. Distinct post-gate merge step |
| *(not η)* | `fragment_consolidation_enabled` | **true** | Whether fragment merge runs | **VERIFIED_FROM_ARTIFACT** |

### Explicit non-correspondences

- Do **not** treat `fragment_centroid_threshold=0.85` as η.
- Do **not** treat `minimum_cross_vehicle_support=1` as η.
- Do **not** treat `dbscan_min_samples=2` as η.
- Do **not** assume peer IEEE default `FinalGnnFleetConfig.min_cluster_size=10` was the historical P7/P8 η.

### γ = 2?

Freeze **value** `minimum_distinct_vehicles=2` is verified. Whether the paper’s γ is exactly that freeze key is **UNRECOVERABLE** from balanced publication sources (no greek mapping in freeze). Candidate mapping only.

### β = 0.5?

Freeze **value** `minimum_campaign_cohesion=0.5` is verified. Ablation docs that cite the publication pair label cohesion as β=0.5. Balanced freeze itself stores the numeric key without printing β.

---

## PR #24 incorrect coupling (removed)

Previous committed code in `PublicationFleetConfig` (pre-correction) contained:

```text
# η for |C_k|: ... freeze has no separate cluster-size key, so η := dbscan_min_samples (=2).
minimum_cluster_size: int = 2  # η := freeze dbscan_min_samples
```

and the gate used:

```text
size >= cfg.minimum_cluster_size
```

with `minimum_cluster_size` defaulting to 2 **because** it was documented as equal to `dbscan_min_samples`.

That coupling is **unsupported** by PR #25’s recovery audit:

- DBSCAN `min_samples=2` is verified.
- Historical η as post-clustering `\|C_k\|` is **unrecoverable**.
- Equality of the two numbers was an invention, not a recovered historical fact.

### Correction

1. Renamed / replaced the gate size field with **`min_campaign_cluster_size: int | None = None`**.
2. Added **`require_min_campaign_cluster_size()`** — raises if unset; **never** falls back to `dbscan_min_samples`.
3. Gate predicate takes an explicit `eta` argument resolved only via that helper.
4. Cluster summary rows now emit both `gate_eta_min_campaign_cluster_size` and `gate_dbscan_min_samples` so accidental equality is observable, not implicit.
5. CTT entry points (`resolve_ctt_fleet_config`, `run_fleet_campaign_inference`, `run_scenario_evaluation`, `scripts/run_ctt_aligned_publication.py`) require η to be passed explicitly for any campaign run.
6. Runner default mode is `write-config-only` until η is decided; smoke/pilot/full refuse to start without `--eta`.

It is now impossible to accidentally use `dbscan_min_samples` as `min_campaign_cluster_size` through defaults or inheritance.

---

## Historical OCSLab / P7 / P8

**Untouched.** No re-run. No claim that any newly chosen η produced:

- Strong F1: `0.533 / 0.733 / 1.000`
- Weak F1: `0.067 / 0.500 / 0.717`

Those remain historical publication artifacts under `experimental-2026-06-23/01_primary_ocslab_balanced/`.

---

## Prospective CTT

Aligned CTT validation **requires** an explicit frozen η **before** examining new CTT results.

- Do not choose η by CTT performance.
- Do not default η to 2 or 10.
- After an independent η decision, pass `--eta N` (alias `--min-campaign-cluster-size`) to the runner.

No CTT experiment was executed as part of this correction.

---

## Tests covering the separation

`tests/test_ctt_aligned_publication_fleet.py`:

1. Changing `dbscan_min_samples` does not set η.
2. Changing η does not change `dbscan_min_samples`.
3. DBSCAN (`run_dbscan`) is invoked before the campaign gate.
4. η is applied only in `_cluster_qualifies_campaign` after clustering.
5. Mutating `attack_type` / `label` / `gt_campaign_id` does not change gate outcomes for fixed geometry.

---

## Final status

**READY_FOR_ETA_DECISION**

- DBSCAN and η are fully decoupled.
- CTT requires η explicitly.
- Historical OCSLab results remain untouched.
- No CTT run has been performed in this correction.
