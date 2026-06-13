"""Phase 4 corrected source pool must have independent benign segments."""

import pandas as pd

from src.utils.paths import resolve_project_root


def test_corrected_pool_has_benign_per_model() -> None:
    path = resolve_project_root() / "new_experiments/final_validated_runs/model_diversity_corrected/manifests/model_diversity_source_pool_corrected.csv"
    if not path.exists():
        return
    pool = pd.read_csv(path)
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        sub = pool[(pool.vehicle_model == vm) & (pool.available_benign_descriptors > 0)]
        assert not sub.empty, vm
        assert int(sub["independent_instance_count"].max()) >= 1
