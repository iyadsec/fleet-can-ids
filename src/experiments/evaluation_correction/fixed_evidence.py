"""Fixed-total-malicious-evidence campaign-size control experiment."""

from __future__ import annotations

from typing import Any

from src.experiments.campaign_analysis_corrected import (
    DEFAULT_BENIGN_PER_BENIGN,
    DEFAULT_DESCRIPTORS_PER_VEHICLE,
    DEFAULT_FLEET_SIZE,
    DescriptorBudget,
)

FIXED_TOTAL_MALICIOUS = 20


def fixed_evidence_budget(campaign_size: int) -> DescriptorBudget:
    if campaign_size <= 0:
        raise ValueError("campaign_size must be positive")
    if FIXED_TOTAL_MALICIOUS % campaign_size != 0:
        raise ValueError(
            f"Cannot distribute {FIXED_TOTAL_MALICIOUS} malicious descriptors across {campaign_size} vehicles"
        )
    mal_per = FIXED_TOTAL_MALICIOUS // campaign_size
    benign_per_attacked = DEFAULT_DESCRIPTORS_PER_VEHICLE - mal_per
    return DescriptorBudget(
        descriptors_per_vehicle=DEFAULT_DESCRIPTORS_PER_VEHICLE,
        malicious_per_attacked=mal_per,
        benign_per_attacked=benign_per_attacked,
        benign_per_benign=DEFAULT_BENIGN_PER_BENIGN,
        total_fleet_size=DEFAULT_FLEET_SIZE,
    )


def validate_fixed_evidence_records(records: Any, campaign_size: int) -> dict[str, Any]:
    budget = fixed_evidence_budget(campaign_size)
    n_mal = int((records["ground_truth_malicious"] == 1).sum())
    n_nodes = len(records)
    n_attacked = int(records.loc[records["scenario_role"] == "coordinated", "scenario_vehicle_id"].nunique())
    return {
        "total_nodes": n_nodes,
        "total_malicious_descriptors": n_mal,
        "malicious_per_attacked": budget.malicious_per_attacked,
        "campaign_size": campaign_size,
        "attacked_vehicles": n_attacked,
        "expected_nodes": budget.expected_total_nodes,
        "expected_malicious": FIXED_TOTAL_MALICIOUS,
        "passed": (
            n_nodes == budget.expected_total_nodes
            and n_mal == FIXED_TOTAL_MALICIOUS
            and n_attacked == campaign_size
        ),
    }
