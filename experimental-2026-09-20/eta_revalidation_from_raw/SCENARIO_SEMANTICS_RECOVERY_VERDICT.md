# SCENARIO_SEMANTICS_RECOVERY_VERDICT.md

```text
SCENARIO_SEMANTICS_PARTIALLY_RECOVERED
```

## Verdict rationale

- The original function **`build_mixed_validation_suite` was never committed** (no defining blob in any reachable git object).
- Substantial **publication scenario semantics** are recoverable from committed **peers**, **FREEZE configs**, and **frozen artifacts** (`validation_manifest.csv`, `campaign_metrics.csv`, master YAML, `coordination_strength.py`).
- Semantics are **not** complete enough to rebuild a bit-identical publication validation suite **without inventing** missing wiring (exact mixed-suite cs-strata generator, `PUBLICATION_SCENARIOS` values, missing runner callees).

---

## Answers required by the task

### 1. Exact source(s) recovered

| Source | Role |
|--------|------|
| `MISSING_MODULES.md` + git pickaxe/`def` search | Proves builder never committed |
| Balanced `runner.py` call site | Provenance of validation → search → TEST |
| `model_diversity_final_tuned/validation_scenarios.py` | Closest committed V0–V4 peer builder |
| `coordination_strength.py` | Authoritative prototype-blend dial |
| `campaign_analysis_corrected.py` | DescriptorBudget, attacked-chunk score bands, TEST campaign builder peer |
| Generic `scenario_registry.py` | S0–S4 intent (not proven = missing PUBLICATION_SCENARIOS) |
| `validation_manifest.csv`, `selection_report.md`, `parameter_search.csv` | Historical validation outputs |
| `frozen_results_reference/campaign_metrics.csv` | Historical TEST geometry / F1 |
| `balanced_master_experiment.yaml`, `final_shared_fleet_configuration.yaml` | Budgets, thresholds, freeze |

### 2. Publication seeds

| Role | Seeds | Evidence label |
|------|-------|----------------|
| **Validation** | `{131, 137, 149, 157, 163, 179, 181, 191, 193, 197}` | **VERIFIED_FROM_CODE** (peer `VALIDATION_SEEDS`) + **VERIFIED_FROM_ARTIFACT** (`validation_manifest.csv`) |
| **TEST / P7–P8** | `{11, 23, 37, 41, 53, 67, 71, 83, 97, 101}` | **VERIFIED_FROM_CODE** (`REQUIRED_SEEDS`) + **VERIFIED_FROM_ARTIFACT** (`campaign_metrics.csv`) |

Stand-in used the **validation** list (MATCH), not the TEST list.

### 3. Campaign sizes

| Role | Sizes | Evidence label |
|------|-------|----------------|
| Validation base V2/V3/V4 | **5** | **VERIFIED_FROM_ARTIFACT** + peer code |
| Validation cs strata rows | **2, 5, 10** (18 stub rows; `descriptor_count` NaN) | **VERIFIED_FROM_ARTIFACT**; generator in missing module → **UNRECOVERABLE** exact code |
| TEST / publication tables | **2, 5, 10** | **VERIFIED_FROM_ARTIFACT** (master YAML + `campaign_metrics`) |

Stand-in: size **5 only** → **MISMATCH** vs full publication TEST grid / missing-suite cs strata.

### 4. Node budget

**10 descriptors per vehicle** — **VERIFIED_FROM_ARTIFACT** / **VERIFIED_FROM_CODE** (`source_windows_per_vehicle`, `DescriptorBudget`).

### 5. Fleet size

**20 vehicles** (200 nodes when 10×20) — **VERIFIED_FROM_ARTIFACT**. Isolated TEST condition uses 20 nodes (1 vehicle × 10) in some rows — separate from full-fleet scenarios.

### 6. S3 construction rule (recovered)

**Evidence-backed reconstruction of intent (peer + registry + freeze):**

1. Select ~campaign_size attacked instances (Hyundai/Kia composition) from validation/test catalog with **strong** local evidence (`anomaly_score ≥ 0.80`).
2. Each attacked vehicle: **5 malicious + 5 benign-on-attacked** descriptors (`_build_attacked_vehicle_chunk`).
3. Fill remaining fleet with benign vehicles to total 20.
4. Assign one shared campaign id; `attack_family_mode=shared`.
5. Apply **`apply_coordination_strength`** toward `compute_campaign_prototype(STRONG_ATTACK_DEFAULT)` with high strength (peer validation uses **1.0**; registry range **0.75–1.0**).
6. Build graph **after** scenario sampling (no cosine-NN campaign sampler).

Exact missing `build_mixed_validation_suite` body: **UNRECOVERABLE**.

### 7. S4 construction rule (recovered)

Same structure as S3 but:

- Weak evidence band: `0.55 ≤ anomaly_score < 0.80`.
- Prototype from `WEAK_ATTACK_DEFAULT`.
- Peer validation still uses blend **strength=1.0** (not 0.35).
- Stand-in used **0.35** → **MISMATCH** vs peer.

Exact missing-suite weak strength schedule: **UNRECOVERABLE** beyond peer/registry bounds.

### 8. Did behavioral similarity influence campaign construction?

**Yes — by design, after sampling.**  
Prototype blend toward a shared attack-family behavioural centroid (`coordination_strength.py`).  
**Not** by selecting cross-vehicle nearest neighbors at τ=0.95.  
Label: mechanism **VERIFIED_FROM_CODE**; stand-in effectiveness failure (cosine≈0.17) **VERIFIED_FROM_ARTIFACT** (diagnostic).

### 9. Mega-clusters explanation

Publication Strong campaigns: mean `largest_cluster_size≈34`, mean `predicted_campaign_size≈5`, rarely `benign_vehicles_included` large.  
Reconstructed S3/S4: often `r_k≈20`, `|C_k|>100`, `c_k≈0.2–0.4`.

→ Mega-clusters are **not** expected publication behavior; they are **caused by reconstructed scenario composition** (weak imposed coordination vs peer: malicious-only mask + weak strength 0.35; flat OEM pooling vs instance catalog), producing insufficient same-GT similarity for selective cross edges and compact clusters.  
Fleet size / τ / DBSCAN freeze **MATCH** publication — not the root mismatch.

### 10. Can a faithful publication-equivalent validation suite be rebuilt **without inventing** methodology?

**No — not yet.**

Recovered peers + artifacts specify much of the **intent**, but:

- original `build_mixed_validation_suite` remains missing;
- validation cs∈{2,5,10} generator for the mixed suite is only evidenced as stub rows;
- `PUBLICATION_SCENARIOS` / TEST orchestration remain missing;
- adopting the peer builder wholesale would still be a **substitution**, not a recovery of the exact publication module.

Safe next step (out of scope here): decide whether to (a) treat peer `build_validation_scenarios` + documented deltas as an explicit prospective protocol, or (b) stop fleet η validation until the missing module is obtained from outside git.

---

## Property labels A–T (summary)

See `SCENARIO_BUILDER_COMPARISON.csv` for stand-in vs publication rows.

| ID | Property | Publication evidence label |
|----|----------|----------------------------|
| A | seeds | VERIFIED_FROM_CODE + ARTIFACT |
| B | campaign sizes | VERIFIED_FROM_ARTIFACT (TEST); validation cs strata partial |
| C | desc/vehicle | VERIFIED_FROM_ARTIFACT |
| D | benign vehicles | VERIFIED_FROM_ARTIFACT / INFERRED budget |
| E | fleet size | VERIFIED_FROM_ARTIFACT |
| F | attack descriptor selection | VERIFIED_FROM_CODE (peer) |
| G | shared attack family | VERIFIED_FROM_CODE (registry/peer) |
| H | shared source traces | VERIFIED_FROM_ARTIFACT (usually distinct traces) |
| I | S3 strong choice | VERIFIED_FROM_CODE (peer) |
| J | S4 weak choice | VERIFIED_FROM_CODE (peer); strength exact for missing suite UNRECOVERABLE |
| K | score ranges enforced | VERIFIED_FROM_CODE (peer chunk) |
| L | sample before graph | VERIFIED_FROM_CODE |
| M | similarity in sampling | VERIFIED_FROM_CODE (no cosine sampler) |
| N | behavioural similarity by design | VERIFIED_FROM_CODE |
| O | S2 unrelated | VERIFIED_FROM_CODE (peer) |
| P | repetitions | VERIFIED_FROM_ARTIFACT |
| Q | vehicle-token logic | VERIFIED_FROM_CODE (peer catalog) |
| R | node budget | VERIFIED_FROM_ARTIFACT |
| S | benign selection | VERIFIED_FROM_CODE (peer) |
| T | prototype/blending | VERIFIED_FROM_CODE |

---

## Constraints honored

- No TEST / CTT / manuscript changes
- No η re-selection or re-sweep
- No parameter tuning
- No new stand-in builder
- Existing η artifacts retained; validity downgraded in `ETA_SELECTION_VALIDITY.md`
