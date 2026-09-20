# CAMPAIGN_BUILDER_PRESERVATION_CHECK.md

**Status:** Implementation repair check — no experiment execution  
**Builder:** `src.experiments.final_shared_configuration.validation_scenarios.build_mixed_validation_suite`

## Verdict on detector pipeline

```text
DETECTOR_PIPELINE_UNCHANGED
```

## Diff scope reviewed

Compared working tree changes for this repair against detector / methodology surfaces:

| Path / component | Changed? |
|------------------|----------|
| `src/features/` (incl. feature extractor / 24-D semantics) | **No** |
| `src/graph/` (graph construction / cosine) | **No** |
| `src/models/` (GraphSAGE) | **No** |
| `src/evaluation/` (campaign decision / metrics consumers) | **No** |
| Isolation Forest methodology code | **No** |
| DBSCAN pipeline code | **No** |
| Algorithm 1 / Algorithm 2 implementations | **No** |
| Paper equations / notation | **No** (manuscript not modified) |
| `theta_weak` / `theta_strong` / τ / γ / η / β definitions | **No** (builder *reads* 0.55/0.80; refuses drift) |

## What was added (experimental evaluation only)

| Path | Role |
|------|------|
| `src/experiments/final_shared_configuration/validation_scenarios.py` | Recovered `build_mixed_validation_suite` + S0–S4 builders |
| `src/experiments/final_shared_configuration/__init__.py` | Package export |
| `recovered_publication_pipeline/src/experiments/final_shared_configuration/` | Same module on historical missing import path |
| `tests/test_campaign_builder_mixed_validation_suite.py` | Construction constraint tests only |

## Guarantees enforced in builder

1. Prototype-blend runs **only** inside experimental S3/S4 construction.  
2. Only `BEHAVIOURAL_FEATURE_COLUMNS` (24-D) may be rewritten.  
3. `anomaly_score`, vehicle identity, GT membership, attack type, trace/window provenance are asserted unchanged across blend.  
4. S0/S1/S2 never receive shared campaign prototype blend.  
5. TEST split / `test_event_ids` cannot enter validation construction.  
6. Builder source does **not** import GraphSAGE / graph / DBSCAN / torch modules.  
7. `experimental_coordination_strength` is an explicit **experimental builder configuration** field — **not** a paper/model symbol; historical numeric value remains `UNRECOVERABLE`.

## Unit tests executed

```text
python3 -m pytest tests/test_campaign_builder_mixed_validation_suite.py -q
→ 20 passed
```

No S0–S4 validation sweep, η evaluation, TEST, or CTT was run.

## Explicit non-changes

- FLEET-GUARD architecture unchanged  
- Algorithms 1 and 2 unchanged  
- Equations unchanged  
- Existing symbol definitions unchanged  
- η remains \(\lvert C_k\rvert \ge \eta\) (not re-interpreted; not equated to DBSCAN `min_samples`)
