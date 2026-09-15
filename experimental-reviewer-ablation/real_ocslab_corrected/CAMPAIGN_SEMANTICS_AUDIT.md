# Campaign Semantics Audit (corrected ablation gate)

**Status:** Publication campaign-generation semantics are **recoverable**.  
**PR #22 score-band + attack_type-only grouping is NOT publication-faithful.**

This audit is evidence-only. No ablation was run from this file.

---

## 1. Recoverable evidence

| Path | What it shows |
|------|----------------|
| `recovered_publication_pipeline/src/experiments/coordination_strength.py` | `compute_campaign_prototype` = mean behavioural vector for one `attack_type`; `apply_coordination_strength` blends `(1−s)·x + s·proto` with bounded noise (`method=prototype_blend_with_bounded_noise`). |
| `recovered_publication_pipeline/src/experiments/campaign_analysis_corrected.py` | Authoritative corrected builder: platform composition → instance catalog → score-band sampling → prototype blend when `coordination_strength > 0`. |
| `recovered_publication_pipeline/src/experiments/campaign_analysis_generator.py` | Same blend hook; defaults `STRONG/WEAK_ATTACK_DEFAULT="malfunction"`; Chevrolet strong → `"fuzzy"`. |
| `recovered_publication_pipeline/src/experiments/vehicle_instance_builder.py` | Disjoint OEM instances from `(vehicle_model, source_file)` segments; score-band filters; maps Hyundai→Sonata, Kia→Soul, Chevrolet→Spark in display names. |
| `recovered_publication_pipeline/src/experiments/model_diversity_final_tuned/validation_scenarios.py` | V3/V4 coordinated campaigns call blend at **`strength=1.0`**. |
| `recovered_publication_pipeline/src/experiments/scenario_registry.py` | S3/S4 `coordination_strength_range=(0.75, 1.0)`; `attack_family_mode="shared"`. |
| `recovered_publication_pipeline/new_experiments/.../configs/balanced_master_experiment.yaml` | `fleet_size: 20`, `source_windows_per_vehicle: 10`, `campaign_sizes: [2,5,10]`, `behavioural_coordination_only: true`, thresholds 0.55/0.8. |
| `src/benchmark/ocslab_publication_baseline.py` | Reconstructs strong/weak via `generate_corrected_campaign_scenario(..., coordination_strength=1.0)`. |
| `experimental-reviewer-ablation/real_ocslab/scenario_builder_real.py` | PR #22: same attack family + score band only; **no** prototype/blend. |

---

## 2. Exact campaign-generation semantics (evidence-backed)

1. **Catalog OEM instances** from held-out windows grouped by `(vehicle_model, source_file)` with contiguous segments (`build_instance_catalog`).
2. **Choose platform composition** for campaign size 5:
   - strong → `{Hyundai:2, Kia:2, Chevrolet:1}`
   - weak → `{Hyundai:3, Kia:2, Chevrolet:0}`
3. **Select fleet:** `campaign_size` attacked instances matching composition + `(20 − campaign_size)` benign-only instances; disjoint window sets.
4. **Per attacked vehicle:** sample score-band malicious windows (`≥0.80` strong; `[0.55, 0.80)` weak) of preferred attack family (default `"malfunction"`; Chevrolet strong → `"fuzzy"`), plus benign fillers on attacked vehicles.
5. **Assign GT:** one shared `ground_truth_campaign_id` for attacked-vehicle coordinated rows.
6. **Enforce behavioural relatedness (critical):**
   - `proto = compute_campaign_prototype(descriptors, attack_type=primary_attack)` on behavioural feature columns.
   - `apply_coordination_strength(..., strength=1.0, target_mask=coordinated)`.
7. **Validate budget:** 20×10 = 200 nodes.

**How members become behaviourally related:** shared family preference + score band + **forced feature convergence to a shared family mean prototype** — not cosine pre-filtering and not random strong-window grouping.

Operating point used by validation / baseline reconstruct: **`coordination_strength = 1.0`**.

---

## 3. Unrecoverable / ambiguous

- Exact `coordination_strength` inside missing `PUBLICATION_SCENARIOS` for frozen Section VII test tables (closest evidence: **1.0**).
- Missing packages for bit-identical Section VII replay (`final_end_to_end_publication_run.*` helpers, balanced descriptors/models).
- Whether all `scenario_role=="coordinated"` rows (including benign-on-attacked) were intentionally blended (code does blend them).
- Chevrolet strong samples `"fuzzy"` while prototype often uses `"malfunction"` (code quirk retained as recovered).
- Bit-identical frozen scenarios cannot be regenerated without missing descriptor artifacts.

---

## 4. Gate decision for corrected ablation

**YES — defensible publication campaign semantics can be established** from committed `coordination_strength.py` + `campaign_analysis_corrected.py` + instance builder + configs.

Corrected ablation **must** apply `compute_campaign_prototype` + `apply_coordination_strength(strength=1.0)`.  
It **must not** reproduce PR #22’s score-only grouping.

**Caveat:** this recreates publication **GT construction semantics**, not bit-identical Section VII numeric tables.
