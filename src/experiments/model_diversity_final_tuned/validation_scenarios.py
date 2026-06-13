"""Build validation-only scenarios V0–V4 (no test overlap)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    DEFAULT_BENIGN_PER_ATTACKED,
    DEFAULT_BENIGN_PER_BENIGN,
    DEFAULT_DESCRIPTORS_PER_VEHICLE,
    DEFAULT_FLEET_SIZE,
    DEFAULT_MALICIOUS_PER_ATTACKED,
    _append_vehicle_chunk,
    _build_attacked_vehicle_chunk,
    _build_membership,
    _sample_n,
    assert_scenario_budget,
    composition_string,
)
from src.experiments.campaign_analysis_generator import _attack_type_for_instance
from src.experiments.campaign_analysis_corrected import STRONG_ATTACK_DEFAULT, WEAK_ATTACK_DEFAULT
from src.experiments.coordination_strength import apply_coordination_strength, compute_campaign_prototype
from src.experiments.data_splits import is_benign_attack_type
from src.experiments.model_diversity_corrected.benign_fleet import BENIGN_FLEET_COMPOSITION, CONTROLLED_ATTACK_MODELS
from src.experiments.model_diversity_corrected.compositions import resolve_corrected_composition
from src.experiments.scenario_registry import SCENARIO_REGISTRY
from src.experiments.vehicle_instance_builder import (
    _segment_rows,
    build_instance_catalog,
    select_instances_with_benign_composition,
    validate_instance_selection,
    validate_scenario_records,
)

VALIDATION_SEEDS = [131, 137, 149, 157, 163, 179, 181, 191, 193, 197]
SCENARIO_TYPES = ("V0", "V1", "V2", "V3", "V4")


def _scenario_hash(records: pd.DataFrame, scenario: str, seed: int) -> str:
    payload = {
        "scenario": scenario,
        "seed": seed,
        "event_ids": sorted(records["event_id"].astype(str).tolist()),
        "traces": sorted(records.get("source_trace", records.get("source_file", pd.Series())).astype(str).unique().tolist()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _test_event_ids(test_runs_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in test_runs_root.rglob("selected_source_records.csv"):
        if "dry_" in path.parent.name:
            continue
        df = pd.read_csv(path, usecols=["event_id"])
        ids.update(df["event_id"].astype(str))
    return ids


def _benign_only_scenario(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    budget: DescriptorBudget,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V0 — heterogeneous benign fleet (20 benign vehicles)."""
    comp = {"Hyundai": 7, "Kia": 7, "Chevrolet": 6}
    rng = np.random.default_rng(seed)
    benign_inst: list[dict[str, Any]] = []
    used: set[str] = set()
    for model, count in comp.items():
        pool = catalog[
            (catalog["vehicle_model"] == model)
            & catalog["attack_types_available"].eq("benign")
            & (catalog["benign_events"] >= budget.benign_per_benign)
            & ~catalog["scenario_vehicle_id"].isin(used)
        ]
        if len(pool) < count:
            raise ValueError(f"V0 insufficient benign {model}: need {count}, have {len(pool)}")
        chosen = pool.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
        benign_inst.extend(chosen.to_dict("records"))
        used.update(chosen["scenario_vehicle_id"].astype(str).tolist())
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []
    for inst in benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
        chunk = _sample_n(ben_pool, budget.descriptors_per_vehicle, rng)
        _append_vehicle_chunk(
            rows, mapping_rows, chunk, inst, "benign", False, "",
            0, "strong", seed,
        )
    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    mapping_df = pd.DataFrame(mapping_rows)
    membership = _build_membership(scenario_df, seed, 0, 0.0, "strong")
    return scenario_df, membership


def _isolated_attack_scenario(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    budget: DescriptorBudget,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V1 — single attacked vehicle, no coordinated campaign."""
    comp = {"Hyundai": 1, "Kia": 0, "Chevrolet": 0}
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    attacked_inst, benign_inst = select_instances_with_benign_composition(
        catalog,
        n_attacked=1,
        benign_model_composition=BENIGN_FLEET_COMPOSITION,
        attack_strength="strong",
        attack_type=STRONG_ATTACK_DEFAULT,
        model_composition=comp,
        seed=seed,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
        min_attack_events=budget.malicious_per_attacked,
        min_benign_on_benign=budget.benign_per_benign,
    )
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []
    for inst in attacked_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        atk_type = _attack_type_for_instance(inst, "strong")
        chunk = _build_attacked_vehicle_chunk(
            seg_df, attack_strength="strong", atk_type=atk_type,
            budget=budget, weak_th=weak_th, strong_th=strong_th, rng=rng,
        )
        _append_vehicle_chunk(rows, mapping_rows, chunk, inst, "isolated", False, "", 1, "strong", seed)
    for inst in benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
        chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
        _append_vehicle_chunk(rows, mapping_rows, chunk, inst, "benign", False, "", 1, "strong", seed)
    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    mapping_df = pd.DataFrame(mapping_rows)
    membership = _build_membership(scenario_df, seed, 1, 0.0, "strong")
    return scenario_df, membership


def _unrelated_attacks_scenario(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    budget: DescriptorBudget,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V2 — five unrelated attacked vehicles (distinct campaign IDs)."""
    comp = {"Hyundai": 3, "Kia": 2, "Chevrolet": 0}
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    attacked_inst, benign_inst = select_instances_with_benign_composition(
        catalog,
        n_attacked=5,
        benign_model_composition=BENIGN_FLEET_COMPOSITION,
        attack_strength="strong",
        attack_type=STRONG_ATTACK_DEFAULT,
        model_composition=comp,
        seed=seed + 17,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
        min_attack_events=budget.malicious_per_attacked,
        min_benign_on_benign=budget.benign_per_benign,
    )
    rng = np.random.default_rng(seed)
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []
    for i, inst in enumerate(attacked_inst):
        seg_df = _segment_rows(descriptors, manifest, inst)
        atk_type = _attack_type_for_instance(inst, "strong")
        chunk = _build_attacked_vehicle_chunk(
            seg_df, attack_strength="strong", atk_type=atk_type,
            budget=budget, weak_th=weak_th, strong_th=strong_th, rng=rng,
        )
        cid = f"INCIDENT-V2-{i}"
        _append_vehicle_chunk(rows, mapping_rows, chunk, inst, "unrelated", False, cid, 5, "strong", seed)
    for inst in benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
        chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
        _append_vehicle_chunk(rows, mapping_rows, chunk, inst, "benign", False, "", 5, "strong", seed)
    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    if len(scenario_df) != budget.expected_total_nodes:
        raise ValueError(f"V2 node count {len(scenario_df)} != {budget.expected_total_nodes}")
    mapping_df = pd.DataFrame(mapping_rows)
    membership = _build_membership(scenario_df, seed, 5, 0.0, "strong")
    return scenario_df, membership


def _coordinated_campaign_scenario(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    attack_strength: str,
    budget: DescriptorBudget,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V3/V4 — strong/weak coordinated Hyundai/Kia campaign (D1-like)."""
    comp, _, _, reason = resolve_corrected_composition(attack_strength, 1, seed)
    if comp is None:
        raise ValueError(reason or "unsupported composition")
    comp = {k: v for k, v in comp.items() if k in ("Hyundai", "Kia")}
    if sum(comp.values()) != 5:
        comp = {"Hyundai": 3, "Kia": 2}
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    attacked_inst, benign_inst = select_instances_with_benign_composition(
        catalog,
        n_attacked=5,
        benign_model_composition=BENIGN_FLEET_COMPOSITION,
        attack_strength=attack_strength,  # type: ignore[arg-type]
        attack_type=STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT,
        model_composition=comp,
        seed=seed,
        weak_threshold=weak_th,
        strong_threshold=strong_th,
        min_attack_events=budget.malicious_per_attacked,
        min_benign_on_benign=budget.benign_per_benign,
    )
    sel_errors = validate_instance_selection(
        attacked_inst, benign_inst, configured_campaign_size=5, configured_model_composition=comp,
    )
    if sel_errors:
        raise ValueError("; ".join(sel_errors))
    rng = np.random.default_rng(seed)
    campaign_id = f"CAMP-VAL-{attack_strength.upper()}"
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []
    for inst in attacked_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        atk_type = _attack_type_for_instance(inst, attack_strength)  # type: ignore[arg-type]
        chunk = _build_attacked_vehicle_chunk(
            seg_df, attack_strength=attack_strength, atk_type=atk_type,  # type: ignore[arg-type]
            budget=budget, weak_th=weak_th, strong_th=strong_th, rng=rng,
        )
        _append_vehicle_chunk(
            rows, mapping_rows, chunk, inst, "coordinated", True, campaign_id, 5, attack_strength, seed,
        )
    for inst in benign_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
        chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
        _append_vehicle_chunk(rows, mapping_rows, chunk, inst, "benign", False, "", 5, attack_strength, seed)
    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    assert_scenario_budget(scenario_df, budget, 5)
    primary_attack = STRONG_ATTACK_DEFAULT if attack_strength == "strong" else WEAK_ATTACK_DEFAULT
    mal_mask = scenario_df["scenario_role"] == "coordinated"
    if mal_mask.any():
        proto = compute_campaign_prototype(descriptors, attack_type=primary_attack)
        scenario_df, _ = apply_coordination_strength(
            scenario_df, strength=1.0, campaign_prototype=proto, target_mask=mal_mask, seed=seed,
        )
    mapping_df = pd.DataFrame(mapping_rows)
    rec_errors = validate_scenario_records(scenario_df, mapping_df, configured_campaign_size=5)
    if rec_errors:
        raise ValueError("; ".join(rec_errors))
    membership = _build_membership(scenario_df, seed, 5, 1.0, attack_strength)
    return scenario_df, membership


def build_validation_scenarios(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    test_runs_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate validation scenario manifest and persist scenario CSVs."""
    budget = DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
        DEFAULT_FLEET_SIZE,
    )
    val_desc = descriptors[descriptors["split"] == "validation"].copy()
    val_manifest = manifest[manifest["split"] == "validation"].copy()
    test_event_ids = _test_event_ids(test_runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    builders = {
        "V0": lambda s: _benign_only_scenario(val_desc, val_manifest, catalog, config, s, budget),
        "V1": lambda s: _isolated_attack_scenario(val_desc, val_manifest, catalog, config, s, budget),
        "V2": lambda s: _unrelated_attacks_scenario(val_desc, val_manifest, catalog, config, s, budget),
        "V3": lambda s: _coordinated_campaign_scenario(val_desc, val_manifest, catalog, config, s, "strong", budget),
        "V4": lambda s: _coordinated_campaign_scenario(val_desc, val_manifest, catalog, config, s, "weak", budget),
    }
    attack_strength_map = {"V0": "benign", "V1": "strong", "V2": "strong", "V3": "strong", "V4": "weak"}

    for scenario in SCENARIO_TYPES:
        for seed in VALIDATION_SEEDS:
            run_id = f"val_{scenario}_seed{seed}"
            try:
                scenario_df, membership = builders[scenario](seed)
                traces = sorted(scenario_df.get("source_trace", scenario_df.get("source_file", "")).astype(str).unique())
                segments = sorted(scenario_df.get("scenario_vehicle_id", scenario_df.get("vehicle_token", "")).astype(str).unique())
                overlap = int(any(e in test_event_ids for e in scenario_df["event_id"].astype(str)))
                sh = _scenario_hash(scenario_df, scenario, seed)
                scenario_df.to_csv(output_dir / f"{run_id}_records.csv", index=False)
                membership.to_csv(output_dir / f"{run_id}_membership.csv", index=False)
                passed = overlap == 0 and len(scenario_df) > 0
                rows.append(
                    {
                        "validation_run_id": run_id,
                        "validation_seed": seed,
                        "scenario": scenario,
                        "attack_strength": attack_strength_map[scenario],
                        "campaign_size": 5 if scenario in ("V2", "V3", "V4") else (1 if scenario == "V1" else 0),
                        "vehicle_composition": composition_string(
                            dict(scenario_df.groupby("vehicle_model")["scenario_vehicle_id"].nunique())
                        ) if "scenario_vehicle_id" in scenario_df.columns else "",
                        "source_trace_ids": "|".join(traces),
                        "source_segment_ids": "|".join(segments),
                        "descriptor_count": len(scenario_df),
                        "scenario_hash": sh,
                        "overlap_with_test": overlap,
                        "validation_passed": passed,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "validation_run_id": run_id,
                        "validation_seed": seed,
                        "scenario": scenario,
                        "attack_strength": attack_strength_map[scenario],
                        "campaign_size": 0,
                        "vehicle_composition": "",
                        "source_trace_ids": "",
                        "source_segment_ids": "",
                        "descriptor_count": 0,
                        "scenario_hash": "",
                        "overlap_with_test": -1,
                        "validation_passed": False,
                        "error": str(exc),
                    }
                )

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(output_dir.parent / "validation_scenario_manifest.csv", index=False)
    return manifest_df


def scenario_expect_campaign(scenario: str) -> bool:
    return scenario in ("V3", "V4")


def scenario_registry_description(scenario: str) -> str:
    mapping = {"V0": "S0_benign_control", "V1": "S1_isolated", "V2": "S2_non_coordinated", "V3": "S3_strong_campaign", "V4": "S4_weak_campaign"}
    key = mapping.get(scenario, "")
    return SCENARIO_REGISTRY[key].description if key in SCENARIO_REGISTRY else ""
