# PROSPECTIVE_CAMPAIGN_PROTOCOL_V1

**Status:** NEW / PROSPECTIVE — design only  
**Decision basis:** `COORDINATION_REBUILD_VERDICT.md` → `REBUILD_ONLY_AS_NEW_PROSPECTIVE_PROTOCOL`  
**Historical S3/S4 coordination strength:** `INSUFFICIENT_EVIDENCE`  
**Historical builder:** unrecoverable (`build_mixed_validation_suite` missing)

This document defines a **new prospective** controlled-campaign construction protocol for the revised FLEET-GUARD evaluation. It does **not** reconstruct historical P7/P8. It does **not** claim causal common-attacker provenance. It does **not** execute any experiment, validation sweep, TEST, CTT, or η selection.

Terminology (required):

- **controlled coordinated campaign construction** / **controlled descriptor-level campaign construction**
- Real CAN observations → real extracted descriptors → **controlled descriptor transformation** → constructed fleet campaign membership

Do **not** call these naturally occurring coordinated attacks.

---

## 0. Research question (preserved)

The experiment must distinguish:

| Axis | Meaning |
|------|---------|
| **LOCAL EVIDENCE STRENGTH** | Anomaly-score band of selected malicious windows (frozen thresholds) |
| **CROSS-VEHICLE COORDINATION STRENGTH** | Controlled blend toward a shared behavioural prototype (parameter \(s\)) |

Definitions:

- **S3** = strong local anomaly evidence + controlled cross-vehicle coordination  
- **S4** = weak local anomaly evidence + controlled cross-vehicle coordination  

Strong/weak are determined **only** from anomaly-score bands. They are **not** defined by coordination strength \(s\).

---

## 1. Protocol label and non-claims

| Claim | Allowed? |
|-------|----------|
| NEW / PROSPECTIVE protocol for revised evaluation | Yes |
| Uses recovered blend equation / clipping / noise rule | Yes (cite recovered code) |
| Uses frozen score bands, fleet budget, campaign sizes from publication config | Yes (cite FREEZE) |
| Reconstructs historical P7/P8 | **No** |
| Recovers historical \(s\) | **No** — candidates only |
| Selects η | **No** — remains unevaluated |
| Naturally occurring coordinated attacks | **No** |
| Causal common attacker | **No** |

---

## 2. Real data basis

Campaign construction **begins** with descriptors extracted from **real labelled malicious CAN windows** from the OCSLab dataset.

Rules:

1. **No synthetic CAN frames** may be generated.  
2. Windows remain real OCSLab traces; only selected behavioural feature cells may be rewritten by the controlled transform.  
3. For every selected malicious (and, if applicable, benign-on-attacked) descriptor, record:

| Field | Required |
|-------|----------|
| `source_vehicle` | Yes |
| `source_trace` | Yes |
| `window_index` | Yes |
| `attack_family` / `attack_type` | Yes |
| `original_anomaly_score` | Yes |
| `original_24d_behavioural_descriptor` | Yes (store for audit; hash in manifest) |

The original descriptor must remain stored (or content-addressed) for audit even after transformation.

Provenance of extraction path (recovered): real frames → windows → `BEHAVIOURAL_FEATURE_COLUMNS` → IsolationForest `anomaly_score` → scenario sampling → optional `apply_coordination_strength`.

---

## 3. Controlled coordination transform (recovered equation; \(s\) not chosen)

Basis: recovered `apply_coordination_strength` / `compute_campaign_prototype` in  
`recovered_publication_pipeline/src/experiments/coordination_strength.py`.

For each target row \(i\) and transformed coordinate set \(C\):

\[
d'_i[C] = (1-s)\, d_i[C] + s\,\tilde{p} + \varepsilon_i
\]

with recovered noise rule:

\[
\sigma = 0.02\,(1-s), \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I_{|C|})
\]

(when \(\sigma = 0\), no noise is added).

Clipping (recovered):

1. Compute per-feature `feat_min` / `feat_max` over the **target subset** behavioural matrix.  
2. Clip prototype: \(\tilde{p} \leftarrow \mathrm{clip}(p, \mathrm{feat\_min}, \mathrm{feat\_max})\).  
3. After blend (+ optional noise), clip \(d'_i[C]\) to the same per-feature bounds.

Method string recorded in recovered provenance: `prototype_blend_with_bounded_noise`.

### Prospective \(s\) candidates (DO NOT evaluate yet)

Because historical \(s\) is **unrecoverable** from P7/P8 artifacts:

\[
s \in \{0.75,\ 1.0\}
\]

These candidates come from the recovered **allowed registry range** for S3/S4 (`coordination_strength_range=(0.75, 1.0)` in `scenario_registry.py`) and peer implementations. **Neither** may be described as the historical P7/P8 value.

Selection of a single \(s\) is deferred to `COORDINATION_STRENGTH_SELECTION_PLAN.md` (validation-only, non-F1 rule). **Do not execute that plan in this design task.**

---

## 4. Exact transformed features \(C\)

**RECOVERED** from `BEHAVIOURAL_FEATURE_COLUMNS` (24-D) as used by `apply_coordination_strength`:

1. `frame_count`  
2. `unique_can_id_count`  
3. `can_id_entropy`  
4. `most_common_can_id_ratio`  
5. `mean_inter_arrival_time`  
6. `std_inter_arrival_time`  
7. `mean_dlc`  
8. `std_dlc`  
9. `byte_mean_0` … `byte_mean_7` (8 features)  
10. `byte_std_0` … `byte_std_7` (8 features)  

Only columns present in the dataframe and listed above are transformed. Do **not** silently transform additional features.

### Untouched fields (must remain unmodified by the coordination transform)

| Field / group | Status | Rationale |
|---------------|--------|-----------|
| `anomaly_score` | **UNTOUCHED** | Not in `BEHAVIOURAL_FEATURE_COLUMNS`; recovered provenance only *records* `original_anomaly_score` |
| Vehicle identity (`scenario_vehicle_id` / `vehicle_token` / model) | **UNTOUCHED** | Identity must not be rewritten |
| GT labels (`ground_truth_malicious`, `ground_truth_campaign_member`, campaign id columns) | **UNTOUCHED** | Membership is assigned by construction, not by blend |
| `attack_type` / attack family metadata | **UNTOUCHED** | Label of source window |
| Trace / window provenance columns | **UNTOUCHED** | Audit trail |
| Benign-background vehicles' descriptors | **UNTOUCHED** | Not in coordination target mask |
| S0 / S1 / S2 malicious descriptors (when \(s=0\) or transform not applied) | **UNTOUCHED** by coordination blend | See §10 |

Derived GNN views computed **after** the transform may change when behavioural inputs change; that is downstream of \(C\), not an additional silent rewrite of stored metadata.

---

## 5. Prototype construction \(\tilde{p}\)

Recovered function: `compute_campaign_prototype(descriptors, attack_type=…)`.

| Element | Classification | Notes |
|---------|----------------|-------|
| Prototype = mean of behavioural columns over rows with matching `attack_type` | **RECOVERED** | Exact code |
| Default attack family string `"malfunction"` for strong and weak | **RECOVERED** | `STRONG_ATTACK_DEFAULT` / `WEAK_ATTACK_DEFAULT` both `"malfunction"` in peers |
| Chevrolet strong override `"fuzzy"` in some generators | **PROSPECTIVE_CHOICE_REQUIRED** | Peer `campaign_analysis_generator._attack_type_for_instance` vs corrected path; freeze for V1 before execution |
| Prototype pool = full split descriptor table passed into builder (not campaign-only mean) | **RECOVERED** (peer call sites) | `compute_campaign_prototype(descriptors, …)` on the pool argument |
| Restrict prototype to validation-split only vs full OCSLab malicious pool | **PROSPECTIVE_CHOICE_REQUIRED** | Peers use the `descriptors` frame supplied by the caller; missing suite does not pin the exact pool boundary for the lost builder |
| Clip prototype to target-subset min/max before blend | **RECOVERED** | Inside `apply_coordination_strength` |
| Separate prototypes for S3 vs S4 beyond score-band sampling | **NOT_APPLICABLE** | Same machinery; difference is eligibility band, not a different prototype equation |
| Historical P7/P8 prototype id | **NOT_APPLICABLE** | Unrecoverable; prospective protocol assigns its own `prototype_id` / hash |

Until the **PROSPECTIVE_CHOICE_REQUIRED** items are frozen in a follow-on design amendment, implementers must not invent undocumented defaults silently.

---

## 6. S3 — strong local evidence + controlled coordination

**S3** = descriptors satisfying the frozen **STRONG** local anomaly criterion, plus shared campaign membership, plus the controlled coordination transform at the (later) frozen prospective \(s\).

### 6.1 Score eligibility — **RECOVERED**

Frozen thresholds from authoritative master YAML (`local_ids.weak_threshold: 0.55`, `local_ids.strong_threshold: 0.8`) and peer masks:

- Strong band: `anomaly_score ≥ 0.80` and labelled malicious (`_strong_band_mask`).

Malicious descriptors placed into S3 attacked-vehicle chunks **must** come from this strong band.

### 6.2 Sampling procedure — **RECOVERED methodology (peer)** / suite missing

High-level recovered procedure (from `campaign_analysis_corrected._build_attacked_vehicle_chunk` + peer validation coordinated builder):

1. Select `campaign_size` attacked vehicle instances and `fleet_size − campaign_size` benign instances under `DescriptorBudget`.  
2. For each attacked instance, sample `malicious_per_attacked` descriptors from the strong band (attack family per §5).  
3. Sample `benign_per_attacked` companion rows on the same attacked instance (peer: prefer remaining weak-band rows for strong campaigns, with fallback).  
4. Assign a **single** shared `GT_campaign_id` to all coordinated-role rows.  
5. Fill remaining fleet slots with benign-only vehicles (`benign_per_benign` descriptors each).  
6. Apply `apply_coordination_strength` with shared \(\tilde{p}\) and prospective \(s\) to the coordination target mask.

Exact instance-catalog selection for the **missing** `build_mixed_validation_suite` is unrecoverable; prospective V1 adopts the **corrected / peer** selection machinery as the transparent substitute and labels it prospective.

### 6.3 Campaign size / descriptors / fleet — see §8–§9

### 6.4 Benign background

`fleet_size − campaign_size` benign vehicles × `10` benign descriptors each. Benign descriptors are **not** blended toward the campaign prototype.

### 6.5 Attack-family handling

Shared family mode for S3 (`attack_family_mode="shared"` in registry). Default family string `"malfunction"` unless the frozen prospective Chevrolet rule says otherwise (§5).

### 6.6 Prototype + coordination

Same recovered blend as §3–§5. **Do not** use a higher \(s\) for S3 than for S4 merely because local evidence is strong. Local strength and coordination strength are orthogonal axes.

---

## 7. S4 — weak local evidence + controlled coordination

**S4** = descriptors satisfying the frozen **WEAK** local anomaly criterion, with **otherwise identical** construction to S3 unless a predeclared methodological reason exists.

### 7.1 Score eligibility — **RECOVERED**

- Weak band: `0.55 ≤ anomaly_score < 0.80` and labelled malicious (`_weak_band_mask`).

### 7.2 Identity with S3

| Component | S4 vs S3 |
|-----------|----------|
| Blend equation / noise / clipping | Identical |
| Candidate \(s\) set | Identical `{0.75, 1.0}` — same frozen choice when selected |
| Fleet budget / campaign sizes | Identical |
| Shared `GT_campaign_id` | Identical pattern |
| Prototype construction | Same function; default family string same in peers |
| Coordination artificially weakened because evidence is weak | **Forbidden** |

Stand-in reconstruction strength `0.35` for “weak” is **not** part of this prospective protocol.

Primary verified difference: **anomaly-score band only**.

---

## 8. Campaign sizes (frozen condition set)

Use the already established campaign-size conditions:

\[
\{2,\ 5,\ 10\} \text{ attacked vehicles}
\]

Provenance: master YAML `scenario.campaign_sizes: [2, 5, 10]`; peer `REQUIRED_SEEDS` / corrected builder.

These sizes must **not** be changed after results are observed. If a prospective constraint later makes one size impossible on the real descriptor inventory, that size must be **predeclared unsupported** before any detection run — not dropped post hoc.

---

## 9. Node budget / fleet size (provenance)

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Descriptors per vehicle | **10** | FREEZE `scenario.source_windows_per_vehicle: 10`; `DescriptorBudget` |
| Fleet size | **20** vehicles | FREEZE `scenario.fleet_size: 20` |
| Maximum fleet graph nodes | **200** | FREEZE `scenario.total_source_windows: 200` = \(20 \times 10\) |
| Malicious per attacked vehicle | **5** | `DEFAULT_MALICIOUS_PER_ATTACKED` |
| Benign-on-attacked per attacked vehicle | **5** | `DEFAULT_BENIGN_PER_ATTACKED` |
| Benign per benign vehicle | **10** | `DEFAULT_BENIGN_PER_BENIGN` |

Do **not** silently substitute another fleet size. Verification of inventory support remains a pre-execution check (not performed in this design task).

---

## 10. Controls S0 / S1 / S2

Aligned with recovered `scenario_registry.py` expectations; construction details follow peer validation builders where available.

### S0 — no malicious campaign

- Zero attacked vehicles; all descriptors benign.  
- `expect_coordinated_campaign=False`.  
- Coordination strength \(s = 0\); **no** common campaign prototype applied.  
- Measures local FPR / false campaign structure under benign fleet traffic.

### S1 — isolated single-vehicle malicious activity

- Exactly **one** attacked vehicle; remaining vehicles benign.  
- Must **not** be interpreted as a coordinated campaign (`expect_coordinated_campaign=False`).  
- Strong local evidence band for the isolated malicious descriptors (registry `evidence_level="strong"`).  
- **No** shared multi-vehicle `GT_campaign_id`; no cross-vehicle prototype blend (`s = 0`).

### S2 — multiple independent malicious incidents (no shared campaign coordination)

- Multiple attacked vehicles with **independent** incident identities.  
- `expect_coordinated_campaign=False`.  
- Registry: `attack_family_mode="distinct_per_vehicle"`; coordination range allows only low values \([0.0, 0.25]\) — prospective V1 sets **\(s = 0\)** for S2 (no shared prototype blend).

**Independence enforcement (prospective V1):**

1. Assign a **distinct** `GT_campaign_id` / incident id per attacked vehicle (peer pattern `INCIDENT-V2-{i}`).  
2. **Do not** compute or apply a **common** campaign prototype across those independent incidents.  
3. Do **not** mark multi-vehicle shared campaign membership.  
4. Prefer distinct attack-family manifestations per vehicle when the catalog supports it (registry mode); if inventory forces reuse of a family string, independence still holds via distinct incident ids and **absence** of shared prototype blend.  
5. Manifest must record `coordination_strength=0` and empty / per-incident `prototype_id` (no shared prototype).

S2 must **not** apply the common campaign prototype across independent incidents.

---

## 11. Coordination strength \(s\) — deferred

See `COORDINATION_STRENGTH_SELECTION_PLAN.md`.

- Candidates: `{0.75, 1.0}`  
- **Do not** select using final TEST data.  
- **Do not** select by maximizing Campaign F1.  
- **Do not** execute the selection rule in this design task.

---

## 12. η remains unselected

Prior η=2 arose from a non-equivalent reconstructed validation scenario and is **PROVISIONAL ONLY**.

Prospective protocol:

- η candidates remain `{2, 3, 5, 10}`  
- **Do not** evaluate them here.  
- After campaign-generation protocol and coordination strength are frozen, η will be selected using **validation only** under `metric_protocol=revised_jaccard_0.5_v3`.

---

## 13. Reproducibility manifest

Every generated campaign must emit rows conforming to `PROSPECTIVE_CAMPAIGN_MANIFEST_SCHEMA.md`, including:

`scenario`, `seed`, `campaign_size`, `vehicle_token`, `source_vehicle`, `source_trace`, `window_index`, `attack_type`, `original_anomaly_score`, `original_descriptor_hash`, `coordination_strength`, `prototype_id`/`hash`, `transformed_descriptor_hash`, `GT_campaign_id`.

---

## 14. Execution status

| Action | This document |
|--------|----------------|
| Design protocol | Done |
| Run campaign generation | **Not authorized** |
| Run validation / TEST / CTT | **Not authorized** |
| Evaluate \(s\) or η | **Not authorized** |
| Modify manuscript | **Not authorized** |

---

## 15. Document set

| File | Role |
|------|------|
| `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1.md` | This protocol |
| `PROSPECTIVE_CAMPAIGN_MANIFEST_SCHEMA.md` | Audit manifest schema |
| `COORDINATION_STRENGTH_SELECTION_PLAN.md` | Validation-only \(s\) rule (unevaluated) |
| `PROSPECTIVE_VS_HISTORICAL_BOUNDARY.md` | Explicit non-equivalence boundary |
