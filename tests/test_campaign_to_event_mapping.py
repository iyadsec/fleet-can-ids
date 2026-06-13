"""Campaign membership must not map 1:1 to malicious events."""

import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
from src.experiments.evaluation_correction.promotion import apply_corrected_event_decisions


def test_not_all_campaign_members_are_malicious():
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(5)],
            "anomaly_score": [0.1] * 5,
            "local_alert": [0] * 5,
            "weak_signal": [0] * 5,
            "attack_type": ["benign"] * 5,
            "ground_truth_malicious": [0] * 5,
            "cluster_id": [0] * 5,
            "vehicles_in_cluster": [5] * 5,
            "behavioral_cohesion": [0.95] * 5,
            "final_decision": [DECISION_COORDINATED] * 5,
            "gnn_campaign_score": [0.8] * 5,
        }
    )
    out = apply_corrected_event_decisions(df, attack_strength="strong", method="fcgnn")
    assert out["predicted_campaign_membership"].sum() == 5
    assert out["predicted_malicious"].sum() == 0
