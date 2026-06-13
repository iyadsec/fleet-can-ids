"""Vehicle model must remain evaluation-only metadata."""

import pandas as pd

from src.utils.paths import resolve_project_root


def test_benign_pool_has_ground_truth_not_prediction() -> None:
    path = resolve_project_root() / "new_experiments/final_validated_runs/model_diversity_corrected/source_pools/all_model_benign_descriptors.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert "ground_truth_malicious" in df.columns
    assert (df.ground_truth_malicious == 0).all()
    assert "vehicle_model" in df.columns
