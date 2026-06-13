from pathlib import Path

import pandas as pd


def test_validation_scenarios_do_not_overlap_test_events():
    root = Path("new_experiments/final_validated_runs/model_diversity_final_tuned")
    manifest = pd.read_csv(root / "validation_scenarios/validation_scenario_manifest.csv")
    assert (manifest["overlap_with_test"] == 0).all()
    assert manifest["validation_passed"].all()
