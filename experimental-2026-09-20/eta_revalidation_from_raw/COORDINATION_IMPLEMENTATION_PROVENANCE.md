# COORDINATION_IMPLEMENTATION_PROVENANCE.md

Forensic audit only. No experiment execution. No code changes.

## Canonical implementation (single definition)

| Field | Value |
|-------|-------|
| file | `recovered_publication_pipeline/src/experiments/coordination_strength.py` |
| original path | `src/experiments/coordination_strength.py` |
| commit (exact blob) | `61a82038ffcac2066d57d524f9305be88dabe212` (**RECOVERED EXACTLY**, `PROVENANCE.md`) |
| functions | `compute_campaign_prototype`, `apply_coordination_strength`, `measure_mean_pairwise_similarity` |
| alternate `def apply_coordination_strength` in repo | **None** (single definition) |

### `compute_campaign_prototype`

| | |
|--|--|
| inputs | `descriptors: DataFrame`, `attack_type: str`, optional `feature_columns` |
| outputs | `np.ndarray` — column-wise **mean** of behavioural features over rows with that `attack_type` |
| publication-related | Yes (used by campaign builders) |
| validation-related | Yes |
| TEST-related | Yes (via corrected/generator peers) |

### `apply_coordination_strength`

| | |
|--|--|
| inputs | `descriptors`, `strength: float`, `campaign_prototype: np.ndarray`, `target_mask: Series`, optional `feature_columns`, `seed: int=42` |
| outputs | `(descriptors_copy, provenance_DataFrame)` |
| method string recorded | `prototype_blend_with_bounded_noise` |

---

## Callers / copies

| file | role | scenarios | strength source | pub / val / TEST |
|------|------|-----------|-----------------|------------------|
| `model_diversity_final_tuned/validation_scenarios.py` `_coordinated_campaign_scenario` | peer validation V3/V4 builder | V3 strong, V4 weak | **hardcoded `1.0`** both | validation peer (not proven = missing `build_mixed_validation_suite`) |
| `campaign_analysis_corrected.py` `generate_corrected_campaign_scenario` | corrected campaign builder | strong/weak campaigns, sizes 2/5/10 | **parameter** `coordination_strength` | publication-related peer; TEST path uses this when called with a strength |
| `campaign_analysis_generator.py` | earlier generator | strong/weak | **parameter** `coordination_strength` | peer / historical |
| `src/benchmark/ocslab_publication_baseline.py` | benchmark reconstruction | `strong_campaign`, `weak_campaign` | **hardcoded `1.0`** for both | publication-baseline reconstruction (not the missing balanced runner) |
| balanced `runner.py` | imports module only to patch `measure_mean_pairwise_similarity` | — | n/a | publication runner (does **not** call `apply_coordination_strength` directly) |
| `scenario_registry.py` | ranges only | S3/S4 `(0.75, 1.0)` | range, not a single value | generic registry; **not** proven = missing `PUBLICATION_SCENARIOS` |
| `experimental-…/reconstruct_validation_pipeline.py` | **stand-in** (inline blend, does not call `apply_coordination_strength`) | S3/S4 | 1.0 / **0.35** | prospective stand-in only |

Stand-in note: reconstruct script reimplements a similar linear blend on `BEHAVIOURAL_FEATURE_COLUMNS` but (a) does not call the recovered function, (b) uses malicious-only mask, (c) uses weak strength 0.35, (d) omits the recovered noise term.

---

## Missing module relationship

```text
build_mixed_validation_suite  [NEVER COMMITTED]
   └── expected to produce validation_manifest.csv (artifact exists)
   └── likely related to peer build_validation_scenarios + helpers
         └── apply_coordination_strength  [RECOVERED]

PUBLICATION_SCENARIOS / _run_single_test  [NEVER COMMITTED]
   └── expected to pass coordination_strength into campaign builder
   └── peer generate_corrected_campaign_scenario(coordination_strength=…)
```

No freeze YAML key stores a numeric `coordination_strength` used by the balanced publication run. No `campaign_metrics.csv` column records the configured strength.
