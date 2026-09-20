# HISTORICAL_SCENARIO_PROVENANCE.md

Forensic recovery only. No builder changes. No re-runs.

## Search summary

| Search | Result |
|--------|--------|
| `def build_mixed_validation_suite` in any commit blob | **0 hits** |
| `git log --all -S'def build_mixed_validation_suite'` | No defining commit |
| Import/call sites | Present in recovered balanced `runner.py` |
| `git fsck --unreachable` | No recovered defining blob |
| Stashes / tags carrying the module | None found |
| Path `**/final_shared_configuration/**` ever added with the function body | **Never** |

**Conclusion:** the function `build_mixed_validation_suite` in package  
`src.experiments.final_shared_configuration.validation_scenarios` was **never committed**.

---

## Provenance graph (artifact ← writer ← runner ← builder ← descriptors)

```text
table_P6 / table_P7 / table_P8
  ← generate_tables (MISSING base + recovered balanced wrapper)
  ← campaign_metrics.csv / strong_summary.csv / weak_summary.csv
  ← _run_single_test / extract_run_metrics / safety_row   [MISSING]
  ← enumerate_test_runs + PUBLICATION_SCENARIOS           [MISSING]
  ← run_refinement_fcgnn / SharedFleetConfiguration       [MISSING]
  ← TEST scenario construction (campaign_analysis_corrected peers + MISSING registry)
  ← descriptors (retrain_local_pipeline MISSING; scaler FREEZE present)

validation_manifest.csv / parameter_search.csv / selection_report.md
  ← joint_parameter_search                                [MISSING]
  ← build_mixed_validation_suite(val_scenarios, …)        [MISSING]
  ← validation descriptors (split=validation)
  ← DescriptorBudget from master YAML

Closest committed peer (NOT proven identical to missing suite):
  model_diversity_final_tuned.validation_scenarios.build_validation_scenarios
    uses: vehicle_instance_builder, campaign_analysis_corrected helpers,
          coordination_strength.apply_coordination_strength
```

### Call site (balanced runner)

`recovered_publication_pipeline/src/experiments/final_end_to_end_publication_run_balanced/runner.py`:

1. Builds `DescriptorBudget` from master YAML (`source_windows_per_vehicle`, `fleet_size`).
2. Monkey-patches `model_diversity_final_tuned.validation_scenarios._test_event_ids`.
3. Calls **`build_mixed_validation_suite(...)`** → writes `validation_manifest.csv`.
4. Calls **`joint_parameter_search(val_scenarios, config)`** → freeze YAML + `selection_report.md`.
5. Enumerates TEST runs via **`enumerate_test_runs` / `PUBLICATION_SCENARIOS` / `REQUIRED_SEEDS`** (missing).
6. Per run: **`_run_single_test`** → rows aggregated to `campaign_metrics.csv` (and aliases).
7. **`generate_tables`** → P6/P7/P8.

Evidence: **VERIFIED_FROM_CODE** (caller) + **VERIFIED_FROM_ARTIFACT** (outputs exist) + **UNRECOVERABLE** (callees).

---

## What was recovered vs missing

| Piece | Status |
|-------|--------|
| `build_mixed_validation_suite` source | **UNRECOVERABLE** (never in git) |
| `PUBLICATION_SCENARIOS` / `enumerate_test_runs` | **UNRECOVERABLE** |
| `joint_parameter_search` / `SharedFleetConfiguration` | **UNRECOVERABLE** |
| `extract_run_metrics` | **UNRECOVERABLE** |
| Peer `build_validation_scenarios` (V0–V4, seeds 131…197, size 5) | **VERIFIED_FROM_CODE** (closest stand-in peer) |
| `apply_coordination_strength` / `compute_campaign_prototype` | **VERIFIED_FROM_CODE** (recovered exactly) |
| `campaign_analysis_corrected` TEST campaign builder | **VERIFIED_FROM_CODE** (peer; sizes 2/5/10) |
| Generic `scenario_registry.SCENARIO_REGISTRY` S0–S4 specs | **VERIFIED_FROM_CODE** (not proven = missing PUBLICATION_SCENARIOS) |
| `validation_manifest.csv` (50 base + 18 cs stubs) | **VERIFIED_FROM_ARTIFACT** |
| Frozen `campaign_metrics.csv` (90 TEST runs) | **VERIFIED_FROM_ARTIFACT** |
| Master YAML node budget / thresholds / campaign_sizes | **VERIFIED_FROM_ARTIFACT** |
| Freezeselection V3/V4 F1 on validation | **VERIFIED_FROM_ARTIFACT** (`selection_report.md`) |

---

## Behavioral-similarity investigation

Publication construction **intentionally** dials behavioural similarity via prototype blending:

- Module: `coordination_strength.py` — `compute_campaign_prototype(attack_type)` + `apply_coordination_strength(strength, …)`.
- Method name in provenance: `prototype_blend_with_bounded_noise`.
- Peer validation V3/V4: **`strength=1.0`** for both strong and weak (`validation_scenarios.py`).
- Generic registry ranges: S3/S4 `(0.75, 1.0)`; S2 `(0.0, 0.25)`.
- Master YAML: `behavioural_coordination_only: true`.
- Blended features: `BEHAVIOURAL_FEATURE_COLUMNS` (frame timing / CAN-ID / byte stats) — **not** a cosine-NN sampler of existing descriptors.
- **No evidence** of selecting cross-vehicle nearest neighbors by cosine ≥ τ as the scenario sampler. Similarity is **imposed after** sampling by blending toward an attack-family prototype.

Evidence labels: blend mechanism **VERIFIED_FROM_CODE**; exact missing-suite strength schedule **INFERRED_FROM_ARTIFACT** / peer (**not** proven identical to missing module).

---

## Mega-cluster investigation

Frozen TEST `campaign_metrics.csv` (Strong Coordinated, n=30):

| Quantity | Publication mean / median / max |
|----------|----------------------------------|
| `largest_cluster_size` | 33.6 / 27 / 83 |
| `predicted_campaign_size` | 5.2 / 5 / 12 |
| `benign_vehicles_included` | 0.63 / 0 / 8 |
| `cross_vehicle_edges` | 165 / 134 / 400 |
| `graph_nodes` | 200 |

Reconstructed validation diagnostic (S3/S4): often `r_k≈20`, `|C_k|>100`, `c_k≈0.2–0.4`.

**Classification:** mega-clusters are **not** expected publication campaign geometry (**B — caused by reconstructed scenario composition**), with possible contribution from incomplete coordination blending relative to the peer (`mal_mask` and strength differences). Not explained by fleet size (MATCH 20×10) alone. Graph/DBSCAN freeze parameters MATCH; therefore over-merge is downstream of weak same-GT similarity in the stand-in inputs.

Evidence: **VERIFIED_FROM_ARTIFACT** (publication metrics) + **VERIFIED_FROM_CODE** (stand-in vs peer differences).
