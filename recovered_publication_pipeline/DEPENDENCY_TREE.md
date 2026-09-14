# Dependency tree — balanced publication runner

Root: `src.experiments.final_end_to_end_publication_run_balanced.runner`

Source tip: `origin/cursor/campaign-clustering` @ `61a82038ffcac2066d57d524f9305be88dabe212`

Status legend:

- **RECOVERED EXACTLY** — historical file extracted under `recovered_publication_pipeline/`
- **ALREADY PRESENT AND HASH MATCH** — same bytes already in current workspace tree
- **INCOMPATIBLE CURRENT VERSION** — same path exists on current branch but hash differs; historical copy kept only under recovery dir
- **MISSING** — never present in any git object; blocks reproduction

```
runner.py
├── campaign_analysis_corrected                              RECOVERED EXACTLY
│   └── (transitive recovered peers: campaign_analysis_generator,
│        coordination_strength, local_descriptor_normalisation,
│        final_gnn_fleet_decision_experiment, campaign_clustering,
│        campaign_detection_experiment, fleet_graph_builder,
│        fleet_similarity_features, gnn_models, feature_extractor,
│        data_splits, vehicle_instance_builder, …)
├── coordinated_campaign_refinement.refinement_pipeline      MISSING  (run_refinement_fcgnn)
├── final_end_to_end_publication_run.descriptor_analysis     MISSING
├── final_end_to_end_publication_run.edge_sensitivity        MISSING
├── final_end_to_end_publication_run.figures                 MISSING
├── final_end_to_end_publication_run.pipeline                MISSING  (retrain_local_pipeline)
├── final_end_to_end_publication_run.runner                  MISSING  (_run_single_test, …)
├── final_end_to_end_publication_run.scenario_registry       MISSING  (PUBLICATION_SCENARIOS, REQUIRED_SEEDS)
├── final_end_to_end_publication_run.statistics              MISSING
├── final_end_to_end_publication_run.vehicle_evaluation      MISSING
├── final_end_to_end_publication_run_balanced.balanced_split RECOVERED EXACTLY
├── final_end_to_end_publication_run_balanced.guard          RECOVERED EXACTLY
├── final_end_to_end_publication_run_balanced.tables         RECOVERED EXACTLY
│   ├── final_end_to_end_publication_run.tables             MISSING
│   └── final_shared_configuration.shared_config           MISSING
├── final_shared_configuration.metrics                     MISSING
├── final_shared_configuration.parameter_search            MISSING
├── final_shared_configuration.shared_config               MISSING
├── final_shared_configuration.validation_scenarios        MISSING
├── fleet_scaler_loader                                    RECOVERED EXACTLY
├── result_writer                                          RECOVERED EXACTLY
├── strong_campaign_extended.metrics                       MISSING  (bootstrap_ci)
├── vehicle_instance_builder                               RECOVERED EXACTLY
└── utils.paths                                            RECOVERED (hash match vs current)
```

Notable peers vs current workspace:

| Path | vs current workspace |
|------|----------------------|
| `src/graph/fleet_graph_builder.py` | INCOMPATIBLE CURRENT VERSION (hash differs) |
| `src/evaluation/campaign_clustering.py` | ALREADY PRESENT AND HASH MATCH |
| `src/models/gnn_models.py` | ALREADY PRESENT AND HASH MATCH |
| `src/graph/fleet_similarity_features.py` | ALREADY PRESENT AND HASH MATCH |

## Closure verdict

**Incomplete.** Fifteen internal modules required by the balanced runner have no recoverable source.
Reproduction is blocked before any seed can execute (`ImportError` on first missing package).
