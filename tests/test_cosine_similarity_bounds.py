"""Cosine similarity diagnostics must stay within [-1, 1]."""

from pathlib import Path

import pandas as pd

from src.utils.paths import resolve_project_root


def test_final_phase4_cosine_bounds() -> None:
    root = resolve_project_root() / "new_experiments/final_validated_runs/model_diversity_final"
    files = list(root.glob("results/**/similarity_diagnostics.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(p) for p in files])
    assert df.within_valid_range.all()
    assert df.minimum.min() >= -1.0 - 1e-6
    assert df.maximum.max() <= 1.0 + 1e-6
