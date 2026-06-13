"""Corrected Phase 4 run metrics sanity checks."""

import pandas as pd

from src.utils.paths import resolve_project_root


def test_corrected_runs_have_200_nodes() -> None:
    path = resolve_project_root() / "new_experiments/final_validated_runs/model_diversity_corrected/results/run_level_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "is_dry_test" in df.columns:
        df = df[df.is_dry_test != True]
    if df.empty or "graph_nodes" not in df.columns:
        return
    assert (df.graph_nodes == 200).all()
