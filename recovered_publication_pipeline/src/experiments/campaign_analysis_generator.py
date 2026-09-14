"""Generate coordinated campaign scenarios for campaign-size and model-diversity experiments."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from src.experiments.coordination_strength import (
    apply_coordination_strength,
    compute_campaign_prototype,
    measure_mean_pairwise_similarity,
)
from src.experiments.data_splits import is_benign_attack_type
from src.experiments.vehicle_instance_builder import (
    _filter_attack_strength,
    _segment_rows,
    select_instances_for_fleet,
    source_trace_name,
    validate_instance_selection,
    validate_scenario_records,
)

ExperimentKind = Literal["campaign_size", "model_diversity"]
AttackStrength = Literal["strong", "weak"]

STRONG_ATTACK_DEFAULT = "malfunction"
WEAK_ATTACK_DEFAULT = "malfunction"
CHEVROLET_STRONG_ATTACK = "fuzzy"


def _attack_type_for_instance(
    instance: dict[str, Any],
    attack_strength: AttackStrength,
) -> str:
    if instance["vehicle_model"] == "Chevrolet" and attack_strength == "strong":
        return CHEVROLET_STRONG_ATTACK
    return STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT


def generate_campaign_analysis_scenario(
    *,
    experiment: ExperimentKind,
    attack_strength: AttackStrength,
    seed: int,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    campaign_size: int,
    coordination_strength: float,
    model_composition: dict[str, int] | None = None,
    total_fleet_size: int = 20,
    max_events_per_vehicle: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build scenario descriptors, vehicle membership mapping, and provenance.

    Returns (scenario_df, vehicle_mapping_df, membership_df).
    """
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    n_benign = total_fleet_size - campaign_size

    attacked_inst, benign_inst = select_instances_for_fleet(
        catalog,
        n_attacked=campaign_size,
        n_benign=n_benign,
        attack_strength=attack_strength,
        attack_type=STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT,
        model_composition=model_composition,
        seed=seed,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
    )
    sel_errors = validate_instance_selection(
        attacked_inst,
        benign_inst,
        configured_campaign_size=campaign_size,
        configured_model_composition=model_composition,
    )
    if sel_errors:
        raise ValueError("; ".join(sel_errors))

    rng = np.random.default_rng(seed)
    campaign_id = f"CAMP-{experiment.upper()}-{attack_strength.upper()}"
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []

    for inst in attacked_inst + benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        is_attacked = inst in attacked_inst
        if is_attacked:
            atk_type = _attack_type_for_instance(inst, attack_strength)
            chunk = _filter_attack_strength(
                seg_df,
                attack_strength,
                atk_type,
                weak_threshold=weak_th,
                strong_threshold=strong_th,
            )
            chunk = chunk.sample(
                n=min(len(chunk), max_events_per_vehicle),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ) if len(chunk) > max_events_per_vehicle else chunk
            role = "coordinated"
        else:
            chunk = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
            chunk = chunk.sample(
                n=min(len(chunk), max_events_per_vehicle),
                random_state=int(rng.integers(0, 2**31 - 1)),
            ) if len(chunk) > max_events_per_vehicle else chunk
            role = "benign"

        if chunk.empty:
            raise RuntimeError(f"No events for instance {inst['scenario_vehicle_id']} role={role}")

        chunk = chunk.copy()
        chunk["scenario_vehicle_id"] = inst["scenario_vehicle_id"]
        chunk["vehicle_token"] = inst["scenario_vehicle_id"]
        chunk["source_trace"] = source_trace_name(inst["source_file"])
        chunk["source_dataset"] = "Car-Hacking Dataset"
        chunk["scenario_role"] = role
        chunk["ground_truth_campaign_id"] = campaign_id if is_attacked else ""
        chunk["ground_truth_campaign_member"] = int(is_attacked)
        chunk["ground_truth_malicious"] = int(is_attacked)
        chunk["scenario_gt_malicious"] = int(is_attacked)
        rows.append(chunk)
        inst_copy = {k: v for k, v in inst.items() if k != "window_ids"}
        wids = inst.get("window_ids", [])
        inst_copy["window_ids"] = "|".join(str(int(w)) for w in wids) if wids else ""
        mapping_rows.append(
            {
                **inst_copy,
                "scenario_role": role,
                "configured_campaign_size": campaign_size,
                "attack_strength": attack_strength,
                "experiment": experiment,
                "seed": seed,
            }
        )

    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    if attack_strength in ("strong", "weak") and coordination_strength > 0:
        mal_mask = scenario_df["scenario_role"] == "coordinated"
        if mal_mask.any():
            primary_attack = (
                STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT
            )
            proto = compute_campaign_prototype(descriptors, attack_type=primary_attack)
            scenario_df, _ = apply_coordination_strength(
                scenario_df,
                strength=coordination_strength,
                campaign_prototype=proto,
                target_mask=mal_mask,
                seed=seed,
            )

    mapping_df = pd.DataFrame(mapping_rows)
    rec_errors = validate_scenario_records(
        scenario_df, mapping_df, configured_campaign_size=campaign_size
    )
    if rec_errors:
        raise ValueError("; ".join(rec_errors))

    membership = []
    for _, r in scenario_df.iterrows():
        membership.append(
            {
                "event_id": r["event_id"],
                "window_id": int(r["window_id"]),
                "scenario_vehicle_id": r["scenario_vehicle_id"],
                "vehicle_token": r.get("vehicle_token", r["scenario_vehicle_id"]),
                "vehicle_model": r["vehicle_model"],
                "source_file": r.get("source_file", ""),
                "source_trace": r.get("source_trace", ""),
                "attack_type": r["attack_type"],
                "split": "test",
                "seed": seed,
                "campaign_size": campaign_size,
                "coordination_strength": coordination_strength,
                "attack_strength": attack_strength,
                "experiment": experiment,
                "scenario_id": f"CA-{experiment}",
                "scenario_role": r["scenario_role"],
                "ground_truth_campaign_id": r.get("ground_truth_campaign_id", ""),
                "ground_truth_campaign_member": int(r.get("ground_truth_campaign_member", 0)),
                "ground_truth_malicious": int(r.get("ground_truth_malicious", 0)),
            }
        )
    membership_df = pd.DataFrame(membership)
    if (scenario_df["scenario_role"] == "coordinated").any():
        sim = measure_mean_pairwise_similarity(
            scenario_df, scenario_df["scenario_role"] == "coordinated"
        )
        membership_df["mean_malicious_pairwise_similarity"] = sim

    scenario_df["experiment"] = experiment
    scenario_df["attack_strength"] = attack_strength
    scenario_df["configured_campaign_size"] = campaign_size
    scenario_df["configured_coordination_strength"] = coordination_strength
    scenario_df["total_fleet_size"] = total_fleet_size
    scenario_df["model_diversity"] = (
        len({i["vehicle_model"] for i in attacked_inst}) if attacked_inst else 0
    )

    max_nodes = int(config.get("graph", {}).get("max_nodes_per_scenario", 1200))
    if len(scenario_df) > max_nodes:
        attacked = scenario_df[scenario_df["scenario_role"] != "benign"]
        benign = scenario_df[scenario_df["scenario_role"] == "benign"]
        n_ben = max(max_nodes - len(attacked), 0)
        benign = benign.sample(
            n=min(len(benign), n_ben),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        scenario_df = pd.concat([attacked, benign], ignore_index=True)

    return scenario_df, mapping_df, membership_df
