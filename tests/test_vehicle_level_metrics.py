"""Vehicle-level metrics include precision and coverage."""

import pandas as pd

from src.experiments.evaluation_correction.metrics import compute_vehicle_detailed_metrics


def test_vehicle_precision_below_perfect_recall():
    events = pd.DataFrame(
        {
            "event_id": ["a1", "a2", "b1", "b2"],
            "scenario_vehicle_id": ["v1", "v1", "v2", "v2"],
            "ground_truth_malicious": [1, 1, 0, 0],
            "ground_truth_campaign_member": [1, 1, 0, 0],
            "predicted_malicious": [1, 0, 1, 0],
            "predicted_campaign_membership": [1, 0, 0, 0],
        }
    )
    membership = events[["scenario_vehicle_id", "ground_truth_campaign_member"]].drop_duplicates()
    membership = membership.rename(columns={"scenario_vehicle_id": "vehicle_token", "ground_truth_campaign_member": "ground_truth_campaign_member"})
    events = events.rename(columns={"scenario_vehicle_id": "vehicle_token"})
    m = compute_vehicle_detailed_metrics(events, membership.assign(ground_truth_campaign_id=["c", ""]))
    assert m["vehicle_recall"] == 1.0
    assert m["vehicle_precision"] < 1.0
    assert m["vehicle_event_coverage_mean"] == 0.5
