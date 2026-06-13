# Final Scenario Results Summary

**Git commit:** `62bf982c2bb1625b86b30451ef14dad55be4a358`
**Config:** `new_experiments/configs/scenario_experiments.yaml`

## Scenarios executed

- **S0_benign_control**: 68 runs
- **S1_isolated**: 74 runs
- **S2_non_coordinated**: 338 runs
- **S3_strong_campaign**: 320 runs
- **S4_weak_campaign**: 372 runs

## Primary S4 results (FCGNN vs Local IDS)

- Δ recall: +0.0227 (FCGNN 0.8879 vs Local 0.8652)
- Δ f1: +0.0241 (FCGNN 0.6589 vs Local 0.6348)
- Δ fpr: +0.0000 (FCGNN 1.0000 vs Local 1.0000)
- Δ weak_malicious_recovered: +12.1978 (FCGNN 446.3551 vs Local 434.1573)
- Δ campaign_detection_rate: +0.5421 (FCGNN 0.5421 vs Local 0.0000)

## Reproduction

```bash
python scripts/run_new_scenario_experiments.py --config new_experiments/configs/scenario_experiments.yaml --all-scenarios
```
