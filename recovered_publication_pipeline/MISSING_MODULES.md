# Missing modules — never present in any git ref

Searched: all local branches, all `origin/*` remotes, tags (none), reflog,
`git rev-list --objects --all`, `git log --all --full-history -- <path>`,
`git grep` for definitions across every commit, GitHub code/search API,
workspace filesystem, stashes (none), LFS (none), releases (unavailable/empty).

**None of the packages below appear as blobs in any commit.** They were imported by
the committed balanced runner and used to produce frozen artifacts, but the source
was never committed (local-only / agent-session-only at generation time).

Exact import paths and symbols are taken from:

`recovered_publication_pipeline/src/experiments/final_end_to_end_publication_run_balanced/runner.py`
and `…/tables.py`.

| Missing module | Imported symbols | Role |
|----------------|------------------|------|
| `src.experiments.final_end_to_end_publication_run.runner` | `_completeness_report`, `_experiment_config`, `_run_single_test`, `_test_event_ids`, `_write_summary` | Core single-test + orchestration helpers reused by balanced runner |
| `src.experiments.final_end_to_end_publication_run.pipeline` | `retrain_local_pipeline` | Retrain Isolation Forest + regenerate descriptors under balanced split |
| `src.experiments.final_end_to_end_publication_run.scenario_registry` | `PUBLICATION_SCENARIOS`, `REQUIRED_SEEDS`, `enumerate_test_runs` | Scenario definitions + seed list + run grid |
| `src.experiments.final_end_to_end_publication_run.descriptor_analysis` | `run_descriptor_analysis` | Descriptor compactness / privacy / scaling tables |
| `src.experiments.final_end_to_end_publication_run.edge_sensitivity` | `run_edge_sensitivity` | Edge-connectivity sweep |
| `src.experiments.final_end_to_end_publication_run.figures` | `generate_figures` | Publication figures P1–P10 |
| `src.experiments.final_end_to_end_publication_run.statistics` | `run_primary_statistics` | Primary statistical tests |
| `src.experiments.final_end_to_end_publication_run.vehicle_evaluation` | `compute_vehicle_level_metrics` | Vehicle-level metrics |
| `src.experiments.final_end_to_end_publication_run.tables` | `generate_tables` (base) | Base table generators wrapped by balanced `tables.py` |
| `src.experiments.final_shared_configuration.shared_config` | `SharedFleetConfiguration` | Frozen shared gate/graph/DBSCAN config object |
| `src.experiments.final_shared_configuration.parameter_search` | `joint_parameter_search` | Validation-only joint parameter search |
| `src.experiments.final_shared_configuration.validation_scenarios` | `build_mixed_validation_suite` | Mixed validation suite builder |
| `src.experiments.final_shared_configuration.metrics` | `extract_run_metrics`, `safety_row` | Run metric extraction + safety row |
| `src.experiments.coordinated_campaign_refinement.refinement_pipeline` | `run_refinement_fcgnn` | Full FCGNN campaign pipeline used per test run |
| `src.experiments.strong_campaign_extended.metrics` | `bootstrap_ci` | Bootstrap CIs for campaign-size tables |

## Also never committed (data / intermediates)

| Artifact | Notes |
|----------|-------|
| `new_experiments/final_end_to_end_publication_run/` (non-balanced `ORIG_ROOT`) | Referenced by runner/guard; only balanced sibling committed |
| Balanced `descriptors/`, `models/`, `processed/window_features.csv` | Required by runner; not in git |
| Per-run dumps under `results/scenario_evaluation/runs/` | Not in git (only aggregates) |
| Raw OCSLab traces at author OneDrive path in master YAML | External; not in repo |

## Closest committed stand-ins (NOT equivalent — do not substitute)

| Missing piece | Closest committed code | Why not equivalent |
|---------------|------------------------|--------------------|
| Shared fleet config + joint search | `model_diversity_final_tuned/` gate search + freeze | Different scenario builder, different gate dataclass fields, not wired to balanced runner |
| Campaign cohesion gate consumer | `compute_cluster_behavioral_cohesion` in `final_gnn_fleet_decision_experiment.py` | File defaults DBSCAN eps=1.2 / min_samples=10; publication freeze is 0.5/2 via missing `SharedFleetConfiguration` |
| FCGNN method | `method_fcgnn.py` | Framework-ablation path, not `run_refinement_fcgnn` |
| Scenario registry | `scenario_registry.py` | Generic; missing package defines `PUBLICATION_SCENARIOS` |
