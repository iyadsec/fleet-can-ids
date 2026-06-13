# Frozen pipeline inputs

Artifacts reused from `model_diversity_final/` without retraining or descriptor regeneration.

## File hashes

| Artifact | Path | SHA256 (16) |
|----------|------|-------------|
| final_split_manifest | `final_validated_runs/model_diversity_final/manifests/final_split_manifest.csv` | `6e267abe0130e132` |
| final_window_split_manifest | `final_validated_runs/model_diversity_final/manifests/final_window_split_manifest.csv` | `416f1d3934c6f6a1` |
| local_model_training_manifest | `final_validated_runs/model_diversity_final/manifests/local_model_training_manifest.csv` | `991b1416b2c28d48` |
| scaler_manifest | `final_validated_runs/model_diversity_final/scalers/scaler_manifest.csv` | `4ba7bb55258444cb` |
| all_descriptors | `final_validated_runs/model_diversity_final/descriptors/all_descriptors.csv` | `f8b7a96e26736d21` |
| validation_descriptors | `final_validated_runs/model_diversity_final/descriptors/validation_descriptors.csv` | `a2addb9688db874d` |
| provisional_campaign_gate | `final_validated_runs/model_diversity_final/configs/final_campaign_gate.yaml` | `6b8c138770ac1644` |
| phase4_config | `final_validated_runs/model_diversity_final/configs/phase4_model_diversity_final.yaml` | `3dc31f931ebef0ce` |

## Local Isolation Forest models

- `if_chevrolet_final_42.joblib`: `e3534ebd09878759`
- `if_hyundai_final_42.joblib`: `1a06f9cd4d7624ff`
- `if_kia_final_42.joblib`: `d33fe6b6a4b0a0fa`

## Fleet scalers

- `fleet_benign_scaler_final.json`: `920c424421de7771`

## Frozen components

- Balanced source-level split manifest
- Per-platform Isolation Forest models (train benign only)
- Local thresholds from validation split
- Train-only scaler fitting
- Regenerated descriptors and anomaly scores
- Graph construction methodology and DBSCAN parameters
- GraphSAGE checkpoints / architecture
- Random seeds: production `[11, 23, 37, 41, 53, 67, 71, 83, 97, 101]`
- Validation seeds: separate from production (see validation_scenario_manifest.csv)

## Integrity policy

No local IDS retraining, descriptor regeneration, or graph-model changes in this tuned run.