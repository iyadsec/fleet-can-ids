"""Build S0–S4 scenarios from held-out local windows with post-hoc IF annotation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    generate_corrected_campaign_scenario,
)
from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
from src.experiments.scenario_generator import generate_scenario_records
from src.experiments.scenario_registry import get_scenario
from src.experiments.vehicle_instance_builder import build_instance_catalog

FLEET_SIZE = 20
WINDOWS_PER_VEHICLE = 10
TOTAL_SOURCE_WINDOWS = 200


def _attach_split(descriptors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns and c in manifest.columns]
    return descriptors.merge(manifest[join_cols + ["split"]].drop_duplicates(), on=join_cols, how="left")


def _annotate_stats(scenario_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    annotated = annotate_fleet_candidates(scenario_df, weak_threshold=weak_th, strong_threshold=strong_th)
    stats = candidate_statistics(annotated)
    stats.update({
        "source_windows": len(annotated),
        "malicious_source_windows": int((annotated["ground_truth_malicious"] == 1).sum()),
        "benign_source_windows": int((annotated["ground_truth_malicious"] == 0).sum()),
        "clearly_benign_filtered": int((~annotated["fleet_candidate"] & (annotated["ground_truth_malicious"] == 0)).sum()),
    })
    return annotated, stats


def build_test_scenario(
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
    """Build final test scenario; annotate IF evidence after window selection."""
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
            descriptors=descriptors,
            manifest=manifest,
            config=config,
            max_events_per_vehicle=WINDOWS_PER_VEHICLE,
            max_benign_events=TOTAL_SOURCE_WINDOWS,
        )
        if len(scenario_df) > TOTAL_SOURCE_WINDOWS:
            scenario_df = scenario_df.sample(n=TOTAL_SOURCE_WINDOWS, random_state=seed).copy()

    annotated, stats = _annotate_stats(scenario_df, config)
    return annotated, membership, stats


def build_validation_campaign_scenario(
    *,
    attack_strength: str,
    campaign_size: int,
    seed: int,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    budget: DescriptorBudget,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    val_catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        manifest,
        weak_threshold=float(config.get("local_ids", {}).get("weak_threshold", 0.55)),
        strong_threshold=float(config.get("local_ids", {}).get("strong_threshold", 0.80)),
        min_windows_per_segment=10,
        target_split="validation",
    )
    desc = _attach_split(descriptors, manifest)
    scenario_df, _, membership, _, _ = generate_corrected_campaign_scenario(
        attack_strength=attack_strength,  # type: ignore[arg-type]
        seed=seed,
        descriptors=desc,
        manifest=manifest,
        catalog=val_catalog,
        config=config,
        campaign_size=campaign_size,
        coordination_strength=1.0,
        budget=budget,
    )
    annotated, stats = _annotate_stats(scenario_df, config)
    return annotated, membership, stats
