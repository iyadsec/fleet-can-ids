"""Fixed-evidence campaign-size budget validation."""

import pytest

from src.experiments.evaluation_correction.fixed_evidence import (
    FIXED_TOTAL_MALICIOUS,
    fixed_evidence_budget,
    validate_fixed_evidence_records,
)


@pytest.mark.parametrize("campaign_size,mal_per", [(2, 10), (5, 4), (10, 2)])
def test_fixed_malicious_budget(campaign_size, mal_per):
    b = fixed_evidence_budget(campaign_size)
    assert b.malicious_per_attacked == mal_per
    assert b.malicious_per_attacked * campaign_size == FIXED_TOTAL_MALICIOUS
    assert b.expected_total_nodes == 200


def test_validate_fixed_evidence_records():
    import pandas as pd

    records = pd.DataFrame(
        {
            "ground_truth_malicious": [1] * 20 + [0] * 180,
            "scenario_role": ["coordinated"] * 20 + ["benign_fleet"] * 180,
            "scenario_vehicle_id": [f"v{i//10}" for i in range(200)],
        }
    )
    records.loc[records.index[:20], "scenario_vehicle_id"] = [f"a{i//10}" for i in range(20)]
    val = validate_fixed_evidence_records(records, 10)
    assert val["total_malicious_descriptors"] == 20
    assert val["total_nodes"] == 200
