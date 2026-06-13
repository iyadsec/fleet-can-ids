from pathlib import Path

import pandas as pd


def test_all_gate_candidates_retained():
    path = Path("new_experiments/final_validated_runs/model_diversity_final_tuned/gate_search/all_gate_candidates.csv")
    df = pd.read_csv(path)
    assert len(df) >= 1000
    assert "constraint_pass" in df.columns
    assert df["selected"].sum() == 1
