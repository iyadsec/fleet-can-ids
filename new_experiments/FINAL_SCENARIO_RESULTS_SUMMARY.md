# Final Scenario Results Summary

**Git commit:** `de0ed484da987d45da4b5bdd7baa6d9c974589a8`
**Config:** `new_experiments/configs/scenario_experiments.yaml`

## Scenarios executed

- **S0_benign_control**: 26 runs
- **S1_isolated**: 32 runs
- **S2_non_coordinated**: 66 runs
- **S3_strong_campaign**: 58 runs
- **S4_weak_campaign**: 70 runs

## Primary S4 results (FCGNN vs Local IDS)

- Δ recall: -0.0455 (FCGNN 0.4545 vs Local 0.5000)
- Δ f1: -0.0251 (FCGNN 0.3142 vs Local 0.3393)
- Δ fpr: +0.0000 (FCGNN 1.0000 vs Local 1.0000)
- Δ weak_malicious_recovered: -4.5455 (FCGNN 185.4545 vs Local 190.0000)
- Δ campaign_detection_rate: +0.0909 (FCGNN 0.0909 vs Local 0.0000)

## Reproduction

```bash
python scripts/run_new_scenario_experiments.py --config new_experiments/configs/scenario_experiments.yaml --all-scenarios
```
