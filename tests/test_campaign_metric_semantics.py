"""Campaign metrics semantics for M1 and recomputation."""

import pandas as pd

from src.experiments.evaluation_correction.metrics import aggregate_corrected_run_metrics, reconstruct_cluster_df
from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED


def test_m1_campaign_metrics_na():
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "ground_truth_malicious": [1],
            "predicted_malicious": [1],
            "local_alert": [1],
            "weak_signal": [0],
            "anomaly_score": [0.9],
            "attack_type": ["malfunction"],
            "vehicle_token": ["v1"],
        }
    )
    membership = pd.DataFrame(
        {
            "event_id": ["e1"],
            "vehicle_token": ["v1"],
            "ground_truth_campaign_member": [1],
            "ground_truth_campaign_id": ["c1"],
        }
    )
    m = aggregate_corrected_run_metrics(
        events, pd.DataFrame(), membership, pd.DataFrame(),
        method="local_ids", seed=1, attack_strength="strong", campaign_size=2,
        coordination_strength=1.0, runtime={}, expect_campaign=True,
    )
    assert m["campaign_metrics_na"] is True
    assert pd.isna(m["campaign_f1"])


def test_campaign_f1_recomputable():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "anomaly_score": [0.9, 0.1],
            "local_alert": [1, 0],
            "weak_signal": [0, 0],
            "final_decision": [DECISION_COORDINATED, DECISION_COORDINATED],
            "cluster_id": [0, 0],
            "vehicles_in_cluster": [2, 2],
            "behavioral_cohesion": [0.9, 0.9],
            "predicted_malicious": [1, 0],
            "ground_truth_malicious": [1, 0],
            "vehicle_token": ["v1", "v2"],
        }
    )
    membership = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "vehicle_token": ["v1", "v2"],
            "ground_truth_campaign_member": [1, 1],
            "ground_truth_campaign_id": ["c1", "c1"],
            "scenario_id": ["S"] * 2,
        }
    )
    cluster_df = reconstruct_cluster_df(events)
    m = aggregate_corrected_run_metrics(
        events, pd.DataFrame(), membership, cluster_df,
        method="fcgnn", seed=1, attack_strength="strong", campaign_size=2,
        coordination_strength=1.0, runtime={}, expect_campaign=True,
    )
    assert "campaign_f1" in m
    assert 0.0 <= m["campaign_f1"] <= 1.0
