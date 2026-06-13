"""Final split must not use test windows for IF training."""

import pandas as pd

from src.utils.paths import resolve_project_root


def test_if_training_no_test_overlap() -> None:
    root = resolve_project_root() / "new_experiments/final_validated_runs/model_diversity_final"
    path = root / "manifests/local_model_training_manifest.csv"
    if not path.exists():
        return
    m = pd.read_csv(path)
    assert (m.test_overlap_count == 0).all()
