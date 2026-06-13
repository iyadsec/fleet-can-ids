"""Generate fixed-budget model-diversity scenarios."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    DEFAULT_BENIGN_PER_ATTACKED,
    DEFAULT_BENIGN_PER_BENIGN,
    DEFAULT_DESCRIPTORS_PER_VEHICLE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_MALICIOUS_PER_ATTACKED,
    STRONG_ATTACK_DEFAULT,
    WEAK_ATTACK_DEFAULT,
    _append_vehicle_chunk,
    _build_attacked_vehicle_chunk,
    _build_membership,
    _sample_n,
    assert_scenario_budget,
    composition_string,
)
from src.experiments.campaign_analysis_generator import _attack_type_for_instance
from src.experiments.coordination_strength import apply_coordination_strength, compute_campaign_prototype, measure_mean_pairwise_similarity
from src.experiments.data_splits import is_benign_attack_type
from src.experiments.vehicle_instance_builder import _segment_rows, select_instances_for_fleet, validate_instance_selection, validate_scenario_records


def generate_model_diversity_scenario(
    *,
    attack_strength: str,
    seed: int,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    model_composition: dict[str, int],
    diversity_level: int,
    campaign_size: int = 5,
    coordination_strength: float = 1.0,
    budget: DescriptorBudget | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], DescriptorBudget]:
    budget = budget or DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
        DEFAULT_FLEET_SIZE,
    )
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    n_benign = budget.total_fleet_size - campaign_size
    comp = dict(model_composition)

    attacked_inst, benign_inst = select_instances_for_fleet(
        catalog,
        n_attacked=campaign_size,
        n_benign=n_benign,
        attack_strength=attack_strength,  # type: ignore[arg-type]
        attack_type=STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT,
        model_composition=comp,
        seed=seed,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
        min_attack_events=budget.malicious_per_attacked,
        min_benign_on_attacked=0,
        min_benign_on_benign=budget.benign_per_benign,
    )
    sel_errors = validate_instance_selection(
        attacked_inst,
        benign_inst,
        configured_campaign_size=campaign_size,
        configured_model_composition=comp,
    )
    if sel_errors:
        raise ValueError("; ".join(sel_errors))

    import numpy as np

    rng = np.random.default_rng(seed)
    campaign_id = f"CAMP-MD-{attack_strength.upper()}-D{diversity_level}"
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []

    for inst in attacked_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        atk_type = _attack_type_for_instance(inst, attack_strength)  # type: ignore[arg-type]
        chunk = _build_attacked_vehicle_chunk(
            seg_df,
            attack_strength=attack_strength,  # type: ignore[arg-type]
            atk_type=atk_type,
            budget=budget,
            weak_th=weak_th,
            strong_th=strong_th,
            rng=rng,
        )
        _append_vehicle_chunk(
            rows, mapping_rows, chunk, inst, "coordinated", True, campaign_id,
            campaign_size, attack_strength, seed,
        )

    for inst in benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
        chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
        _append_vehicle_chunk(
            rows, mapping_rows, chunk, inst, "benign", False, "",
            campaign_size, attack_strength, seed,
        )

    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    if len(scenario_df) != budget.expected_total_nodes:
        raise ValueError(f"Node count {len(scenario_df)} != expected {budget.expected_total_nodes}")

    assert_scenario_budget(scenario_df, budget, campaign_size)

    if coordination_strength > 0:
        mal_mask = scenario_df["scenario_role"] == "coordinated"
        if mal_mask.any():
            primary_attack = STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT
            proto = compute_campaign_prototype(descriptors, attack_type=primary_attack)
            scenario_df, _ = apply_coordination_strength(
                scenario_df,
                strength=coordination_strength,
                campaign_prototype=proto,
                target_mask=mal_mask,
                seed=seed,
            )

    mapping_df = pd.DataFrame(mapping_rows)
    rec_errors = validate_scenario_records(scenario_df, mapping_df, configured_campaign_size=campaign_size)
    if rec_errors:
        raise ValueError("; ".join(rec_errors))

    membership = _build_membership(scenario_df, seed, campaign_size, coordination_strength, attack_strength)
    membership["experiment"] = "model_diversity"
    membership["diversity_level"] = diversity_level
    membership["composition_label"] = composition_string(comp)

    if (scenario_df["scenario_role"] == "coordinated").any():
        sim = measure_mean_pairwise_similarity(scenario_df, scenario_df["scenario_role"] == "coordinated")
        membership["mean_malicious_pairwise_similarity"] = sim

    scenario_df["experiment"] = "model_diversity"
    scenario_df["attack_strength"] = attack_strength
    scenario_df["configured_campaign_size"] = campaign_size
    scenario_df["configured_coordination_strength"] = coordination_strength
    scenario_df["total_fleet_size"] = budget.total_fleet_size
    scenario_df["descriptors_per_vehicle"] = budget.descriptors_per_vehicle
    scenario_df["diversity_level"] = diversity_level
    scenario_df["model_diversity"] = diversity_level
    scenario_df["composition_label"] = composition_string(comp)

    mapping_df["experiment"] = "model_diversity"
    mapping_df["diversity_level"] = diversity_level
    mapping_df["composition_label"] = composition_string(comp)

    return scenario_df, mapping_df, membership, comp, budget
