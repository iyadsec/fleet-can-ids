"""Test scenario builder with split attached for all scenario types."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.campaign_analysis_corrected import DescriptorBudget, generate_corrected_campaign_scenario
from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
from src.experiments.final_shared_configuration.source_windows import _annotate_stats, _attach_split
from src.experiments.scenario_generator import generate_scenario_records
from src.experiments.scenario_registry import get_scenario

TOTAL_SOURCE_WINDOWS = 200
WINDOWS_PER_VEHICLE = 10


def build_e2e_test_scenario(
    scenario_key: str,
    *,
    seed: int,
    campaign_size: int,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    budget: DescriptorBudget,
    coordination_strength: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = get_scenario(scenario_key)
    desc = _attach_split(descriptors, manifest)

    if spec.scenario_id in ("S3", "S4"):
        strength = "strong" if spec.scenario_id == "S3" else "weak"
        scenario_df, _, membership, _, _ = generate_corrected_campaign_scenario(
            attack_strength=strength,
            seed=seed,
            descriptors=desc,
            manifest=manifest,
            catalog=catalog,
            config=config,
            campaign_size=campaign_size,
            coordination_strength=coordination_strength,
            budget=budget,
        )
    else:
        scenario_df, membership = generate_scenario_records(
            spec,
            seed=seed,
            campaign_size=campaign_size,
            coordination_strength=coordination_strength,
            descriptors=desc,
            manifest=manifest,
            config=config,
            max_events_per_vehicle=WINDOWS_PER_VEHICLE,
            max_benign_events=TOTAL_SOURCE_WINDOWS,
        )
        if len(scenario_df) > TOTAL_SOURCE_WINDOWS:
            scenario_df = scenario_df.sample(n=TOTAL_SOURCE_WINDOWS, random_state=seed).copy()

    annotated, stats = _annotate_stats(scenario_df, config)
    return annotated, membership, stats
