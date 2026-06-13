"""Tests for corrected event decision logic."""

import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED, DECISION_ISOLATED
from src.experiments.evaluation_correction.promotion import PromotionConfig, apply_corrected_event_decisions


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "e1",
                "anomaly_score": 0.9,
                "local_alert": 1,
                "weak_signal": 0,
                "attack_type": "malfunction",
                "ground_truth_malicious": 1,
                "cluster_id": 0,
                "vehicles_in_cluster": 3,
                "behavioral_cohesion": 0.9,
                "final_decision": DECISION_COORDINATED,
                "gnn_campaign_score": 0.7,
            },
            {
                "event_id": "e2",
                "anomaly_score": 0.1,
                "local_alert": 0,
                "weak_signal": 0,
                "attack_type": "benign",
                "ground_truth_malicious": 0,
                "cluster_id": 0,
                "vehicles_in_cluster": 3,
                "behavioral_cohesion": 0.9,
                "final_decision": DECISION_COORDINATED,
                "gnn_campaign_score": 0.2,
            },
            {
                "event_id": "e3",
                "anomaly_score": 0.6,
                "local_alert": 0,
                "weak_signal": 1,
                "attack_type": "malfunction",
                "ground_truth_malicious": 1,
                "cluster_id": 1,
                "vehicles_in_cluster": 1,
                "behavioral_cohesion": 0.5,
                "final_decision": DECISION_ISOLATED,
                "gnn_campaign_score": 0.4,
            },
        ]
    )


def test_campaign_membership_does_not_force_benign_malicious():
    out = apply_corrected_event_decisions(
        _base_frame(), attack_strength="strong", method="fcgnn", cfg=PromotionConfig()
    )
    benign = out[out["event_id"] == "e2"].iloc[0]
    assert benign["predicted_campaign_membership"] == 1
    assert benign["predicted_malicious"] == 0


def test_strong_local_predicts_without_campaign():
    out = apply_corrected_event_decisions(
        _base_frame(), attack_strength="strong", method="fcgnn", cfg=PromotionConfig()
    )
    strong = out[out["event_id"] == "e1"].iloc[0]
    assert strong["predicted_malicious"] == 1


def test_both_positive_and_negative_predictions_exist():
    out = apply_corrected_event_decisions(
        _base_frame(), attack_strength="weak", method="descriptor_clustering", cfg=PromotionConfig()
    )
    assert out["predicted_malicious"].nunique() >= 2
