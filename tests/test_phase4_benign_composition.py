"""Fixed benign fleet composition."""

from src.experiments.model_diversity_corrected.benign_fleet import BENIGN_FLEET_COMPOSITION


def test_benign_fleet_is_balanced() -> None:
    assert sum(BENIGN_FLEET_COMPOSITION.values()) == 15
    assert BENIGN_FLEET_COMPOSITION == {"Hyundai": 5, "Kia": 5, "Chevrolet": 5}
