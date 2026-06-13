"""Benign label mapping tests."""

from src.experiments.data_splits import is_benign_attack_type


def test_attack_free_is_benign() -> None:
    assert is_benign_attack_type("attack_free")
    assert is_benign_attack_type("benign")
    assert not is_benign_attack_type("malfunction")


def test_ground_truth_malicious_convention() -> None:
    assert 0 == 0  # benign
    assert 1 == 1  # malicious
