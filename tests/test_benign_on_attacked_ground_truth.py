"""Ground-truth malicious windows must never be relabelled benign."""

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type


def test_benign_selection_requires_gt_zero():
    rows = pd.DataFrame(
        {
            "ground_truth_malicious": [0, 1, 0],
            "attack_type": ["malfunction", "malfunction", "benign"],
            "scenario_role": ["coordinated"] * 3,
        }
    )
    benign_candidates = rows[rows["ground_truth_malicious"] == 0]
    assert len(benign_candidates) == 2
    assert not (benign_candidates["ground_truth_malicious"] == 1).any()


def test_weak_score_does_not_imply_benign_gt():
    row = pd.Series({"anomaly_score": 0.2, "ground_truth_malicious": 1, "attack_type": "malfunction"})
    assert row["ground_truth_malicious"] == 1
    assert not is_benign_attack_type(str(row["attack_type"]))
