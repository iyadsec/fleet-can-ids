"""Descriptive publication scenario definitions (no S0–S4 labels in artifacts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.experiments.scenario_registry import SCENARIO_REGISTRY, ScenarioSpec

TestCondition = Literal[
    "Benign-Fleet Control",
    "Isolated Single-Vehicle Attack",
    "Unrelated Multi-Vehicle Incidents",
    "Strong Coordinated Campaign",
    "Weak Coordinated Campaign",
]

INTERNAL_KEYS = {
    "benign_fleet": "S0_benign_control",
    "isolated_attack": "S1_isolated",
    "unrelated_incidents": "S2_non_coordinated",
    "strong_campaign": "S3_strong_campaign",
    "weak_campaign": "S4_weak_campaign",
}

PUBLICATION_NAMES: dict[str, TestCondition] = {
    "benign_fleet": "Benign-Fleet Control",
    "isolated_attack": "Isolated Single-Vehicle Attack",
    "unrelated_incidents": "Unrelated Multi-Vehicle Incidents",
    "strong_campaign": "Strong Coordinated Campaign",
    "weak_campaign": "Weak Coordinated Campaign",
}


@dataclass(frozen=True)
class PublicationScenarioSpec:
    key: str
    test_condition: TestCondition
    internal_key: str
    expect_coordinated_campaign: bool
    attack_strength: str
    uses_campaign_size: bool

    @property
    def internal_spec(self) -> ScenarioSpec:
        return SCENARIO_REGISTRY[self.internal_key]


PUBLICATION_SCENARIOS: dict[str, PublicationScenarioSpec] = {
    "benign_fleet": PublicationScenarioSpec(
        key="benign_fleet",
        test_condition="Benign-Fleet Control",
        internal_key="S0_benign_control",
        expect_coordinated_campaign=False,
        attack_strength="n/a",
        uses_campaign_size=False,
    ),
    "isolated_attack": PublicationScenarioSpec(
        key="isolated_attack",
        test_condition="Isolated Single-Vehicle Attack",
        internal_key="S1_isolated",
        expect_coordinated_campaign=False,
        attack_strength="n/a",
        uses_campaign_size=False,
    ),
    "unrelated_incidents": PublicationScenarioSpec(
        key="unrelated_incidents",
        test_condition="Unrelated Multi-Vehicle Incidents",
        internal_key="S2_non_coordinated",
        expect_coordinated_campaign=False,
        attack_strength="n/a",
        uses_campaign_size=False,
    ),
    "strong_campaign": PublicationScenarioSpec(
        key="strong_campaign",
        test_condition="Strong Coordinated Campaign",
        internal_key="S3_strong_campaign",
        expect_coordinated_campaign=True,
        attack_strength="strong",
        uses_campaign_size=True,
    ),
    "weak_campaign": PublicationScenarioSpec(
        key="weak_campaign",
        test_condition="Weak Coordinated Campaign",
        internal_key="S4_weak_campaign",
        expect_coordinated_campaign=True,
        attack_strength="weak",
        uses_campaign_size=True,
    ),
}

SAFETY_SCENARIO_KEYS = ("benign_fleet", "isolated_attack", "unrelated_incidents")
CAMPAIGN_SCENARIO_KEYS = ("strong_campaign", "weak_campaign")
CAMPAIGN_SIZES = (2, 5, 10)
REQUIRED_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101)


def enumerate_test_runs(*, dry_run: bool = False) -> list[dict]:
    seeds = [REQUIRED_SEEDS[0]] if dry_run else list(REQUIRED_SEEDS)
    runs: list[dict] = []
    for key in PUBLICATION_SCENARIOS:
        spec = PUBLICATION_SCENARIOS[key]
        sizes = CAMPAIGN_SIZES if spec.uses_campaign_size else [0]
        for seed in seeds:
            for cs in sizes:
                runs.append(
                    {
                        "scenario_key": key,
                        "test_condition": spec.test_condition,
                        "internal_key": spec.internal_key,
                        "seed": seed,
                        "campaign_size": cs,
                        "attack_strength": spec.attack_strength,
                        "expect_campaign": spec.expect_coordinated_campaign,
                    }
                )
    return runs
