"""Corrected Phase 3 campaign-size scenarios with fixed descriptor budgets and platform composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.experiments.campaign_analysis_generator import (
    CHEVROLET_STRONG_ATTACK,
    STRONG_ATTACK_DEFAULT,
    WEAK_ATTACK_DEFAULT,
    _attack_type_for_instance,
)
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

# Re-export for tests
__all__ = [
    "DescriptorBudget",
    "audit_descriptor_budget",
    "generate_corrected_campaign_scenario",
    "platform_composition",
]

AttackStrength = Literal["strong", "weak"]
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

DEFAULT_DESCRIPTORS_PER_VEHICLE = 10
DEFAULT_MALICIOUS_PER_ATTACKED = 5
DEFAULT_BENIGN_PER_ATTACKED = 5
DEFAULT_BENIGN_PER_BENIGN = 10
DEFAULT_FLEET_SIZE = 20


@dataclass(frozen=True)
class DescriptorBudget:
    descriptors_per_vehicle: int
    malicious_per_attacked: int
    benign_per_attacked: int
    benign_per_benign: int
    total_fleet_size: int = DEFAULT_FLEET_SIZE

    @property
    def expected_total_nodes(self) -> int:
        return self.total_fleet_size * self.descriptors_per_vehicle


STRONG_CS2_ROTATIONS: list[dict[str, int]] = [
    {"Hyundai": 1, "Kia": 1, "Chevrolet": 0},
    {"Hyundai": 1, "Kia": 0, "Chevrolet": 1},
    {"Hyundai": 0, "Kia": 1, "Chevrolet": 1},
]


def platform_composition(
    attack_strength: AttackStrength,
    campaign_size: int,
    seed: int,
) -> dict[str, int]:
    """Approved attacked-vehicle platform counts (sum equals campaign_size)."""
    if attack_strength == "strong":
        if campaign_size == 2:
            idx = REQUIRED_SEEDS.index(seed) if seed in REQUIRED_SEEDS else seed % 3
            return dict(STRONG_CS2_ROTATIONS[idx % len(STRONG_CS2_ROTATIONS)])
        if campaign_size == 5:
            return {"Hyundai": 2, "Kia": 2, "Chevrolet": 1}
        if campaign_size == 10:
            return {"Hyundai": 4, "Kia": 3, "Chevrolet": 3}
    else:
        if campaign_size == 2:
            return {"Hyundai": 1, "Kia": 1, "Chevrolet": 0}
        if campaign_size == 5:
            return {"Hyundai": 3, "Kia": 2, "Chevrolet": 0}
        if campaign_size == 10:
            return {"Hyundai": 5, "Kia": 5, "Chevrolet": 0}
    raise ValueError(f"Unsupported campaign_size={campaign_size}")


def composition_string(comp: dict[str, int]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(comp.items()) if v > 0]
    return ", ".join(parts) if parts else "none"


def _malicious_attacked_pool(
    catalog: pd.DataFrame,
    attack_strength: AttackStrength,
    budget: DescriptorBudget,
    *,
    vehicle_model: str | None = None,
) -> pd.DataFrame:
    atk_col = "strong_attack_events" if attack_strength == "strong" else "weak_attack_events"
    pool = catalog[
        (~catalog["attack_types_available"].eq("benign"))
        & (catalog[atk_col] >= budget.malicious_per_attacked)
    ]
    if vehicle_model:
        pool = pool[pool["vehicle_model"] == vehicle_model]
    return pool


def _benign_fleet_pool(
    catalog: pd.DataFrame,
    budget: DescriptorBudget,
    *,
    exclude_ids: set[str],
) -> pd.DataFrame:
    return catalog[
        catalog["attack_types_available"].eq("benign")
        & (catalog["benign_events"] >= budget.benign_per_benign)
        & ~catalog["scenario_vehicle_id"].isin(exclude_ids)
    ]


def _weak_band_mask(df: pd.DataFrame, weak_th: float, strong_th: float) -> pd.Series:
    mal = ~df["attack_type"].map(is_benign_attack_type)
    score = pd.to_numeric(df["anomaly_score"], errors="coerce")
    return mal & (score >= weak_th) & (score < strong_th)


def _strong_band_mask(df: pd.DataFrame, strong_th: float) -> pd.Series:
    mal = ~df["attack_type"].map(is_benign_attack_type)
    score = pd.to_numeric(df["anomaly_score"], errors="coerce")
    return mal & (score >= strong_th)


def _build_attacked_vehicle_chunk(
    seg_df: pd.DataFrame,
    *,
    attack_strength: AttackStrength,
    atk_type: str,
    budget: DescriptorBudget,
    weak_th: float,
    strong_th: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
  Partition one disjoint malicious segment into malicious + benign-on-attacked descriptors.

  Benign-on-attacked rows use ground_truth_malicious=0 while remaining campaign members.
  """
    if attack_strength == "strong":
        mal_pool = _filter_attack_strength(
            seg_df, "strong", atk_type, weak_threshold=weak_th, strong_threshold=strong_th
        )
        mal_chunk = _sample_n(mal_pool, budget.malicious_per_attacked, rng)
        remaining = seg_df.drop(mal_chunk.index, errors="ignore")
        ben_pool = remaining[_weak_band_mask(remaining, weak_th, strong_th)]
        if len(ben_pool) < budget.benign_per_attacked:
            ben_pool = remaining[~remaining.index.isin(mal_chunk.index)]
        ben_chunk = _sample_n(ben_pool, budget.benign_per_attacked, rng)
        ben_chunk = ben_chunk.copy()
        ben_chunk["ground_truth_malicious"] = 0
        mal_chunk = mal_chunk.copy()
        mal_chunk["ground_truth_malicious"] = 1
        return pd.concat([mal_chunk, ben_chunk], ignore_index=True)

    weak_pool = _filter_attack_strength(
        seg_df, "weak", atk_type, weak_threshold=weak_th, strong_threshold=strong_th
    )
    if len(weak_pool) < budget.malicious_per_attacked:
        raise ValueError(
            f"Weak segment has {len(weak_pool)} weak events; need {budget.malicious_per_attacked}"
        )
    mal_chunk = _sample_n(weak_pool, budget.malicious_per_attacked, rng)
    remaining = seg_df.drop(mal_chunk.index, errors="ignore")
    ben_pool = remaining[~remaining.index.isin(mal_chunk.index)]
    ben_chunk = _sample_n(ben_pool, budget.benign_per_attacked, rng)
    mal_chunk = mal_chunk.copy()
    mal_chunk["ground_truth_malicious"] = 1
    ben_chunk = ben_chunk.copy()
    ben_chunk["ground_truth_malicious"] = 0
    return pd.concat([mal_chunk, ben_chunk], ignore_index=True)


def audit_descriptor_budget(
    catalog: pd.DataFrame,
    *,
    campaign_sizes: list[int] | None = None,
    seeds: list[int] | None = None,
) -> tuple[DescriptorBudget, dict[str, Any]]:
    """Verify fixed budget is achievable for all strength × size × seed compositions."""
    campaign_sizes = campaign_sizes or [2, 5, 10]
    seeds = seeds or REQUIRED_SEEDS
    failures: list[str] = []
    per_condition: dict[str, bool] = {}

    def _try_budget(budget: DescriptorBudget) -> list[str]:
        errs: list[str] = []
        for strength in ("strong", "weak"):
            for cs in campaign_sizes:
                for seed in seeds:
                    comp = platform_composition(strength, cs, seed)  # type: ignore[arg-type]
                    key = f"{strength}_cs{cs}_seed{seed}"
                    try:
                        _assert_catalog_supports(
                            catalog,
                            comp,
                            n_benign=DEFAULT_FLEET_SIZE - cs,
                            attack_strength=strength,  # type: ignore[arg-type]
                            budget=budget,
                        )
                        per_condition[key] = True
                    except ValueError as exc:
                        per_condition[key] = False
                        errs.append(f"{key}: {exc}")
        return errs

    preferred = DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
    )
    failures = _try_budget(preferred)
    chosen = preferred

    summary = {
        "chosen_budget": {
            "descriptors_per_vehicle": chosen.descriptors_per_vehicle,
            "malicious_per_attacked": chosen.malicious_per_attacked,
            "benign_per_attacked": chosen.benign_per_attacked,
            "benign_per_benign": chosen.benign_per_benign,
            "total_fleet_size": chosen.total_fleet_size,
        },
        "expected_total_nodes": chosen.expected_total_nodes,
        "preferred_budget_supported": (
            chosen.descriptors_per_vehicle == DEFAULT_DESCRIPTORS_PER_VEHICLE
            and not failures
        ),
        "failures": failures,
        "per_condition": per_condition,
        "campaign_sizes_supported": {
            cs: all(
                per_condition.get(f"{s}_cs{cs}_seed{seed}", False)
                for s in ("strong", "weak")
                for seed in seeds
            )
            for cs in campaign_sizes
        },
    }
    if failures:
        raise ValueError(
            f"Descriptor budget audit failed ({len(failures)} conditions). "
            f"First: {failures[0]}"
        )
    return chosen, summary


def _assert_catalog_supports(
    catalog: pd.DataFrame,
    model_composition: dict[str, int],
    *,
    n_benign: int,
    attack_strength: AttackStrength,
    budget: DescriptorBudget,
) -> None:
    for model, count in model_composition.items():
        if count <= 0:
            continue
        mal_n = len(_malicious_attacked_pool(catalog, attack_strength, budget, vehicle_model=model))
        if mal_n < count:
            raise ValueError(
                f"{model} {attack_strength}: need {count} malicious segments "
                f"(mal>={budget.malicious_per_attacked}), pool={mal_n}"
            )
    ben_pool = _benign_fleet_pool(catalog, budget, exclude_ids=set())
    if len(ben_pool) < n_benign:
        raise ValueError(
            f"Need {n_benign} benign fleet instances (ben>={budget.benign_per_benign}), pool={len(ben_pool)}"
        )


def _sample_n(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Insufficient rows: need {n}, have {len(df)}")
    if len(df) == n:
        return df.copy()
    return df.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1)))


def assert_scenario_budget(scenario_df: pd.DataFrame, budget: DescriptorBudget, campaign_size: int) -> None:
    assert scenario_df["scenario_vehicle_id"].nunique() == budget.total_fleet_size
    attacked_instances = scenario_df.loc[
        scenario_df["ground_truth_campaign_member"] == 1, "scenario_vehicle_id"
    ].nunique()
    assert attacked_instances == campaign_size

    vehicle_counts = scenario_df.groupby("scenario_vehicle_id").size()
    assert vehicle_counts.nunique() == 1
    assert int(vehicle_counts.iloc[0]) == budget.descriptors_per_vehicle

    malicious_counts = (
        scenario_df[scenario_df["ground_truth_malicious"] == 1].groupby("scenario_vehicle_id").size()
    )
    attacked_ids = scenario_df.loc[
        scenario_df["ground_truth_campaign_member"] == 1, "scenario_vehicle_id"
    ].unique()
    mal_attacked = malicious_counts.reindex(attacked_ids).dropna()
    assert mal_attacked.nunique() == 1
    assert int(mal_attacked.iloc[0]) == budget.malicious_per_attacked

    benign_attacked = (
        scenario_df[
            (scenario_df["ground_truth_campaign_member"] == 1)
            & (scenario_df["ground_truth_malicious"] == 0)
        ]
        .groupby("scenario_vehicle_id")
        .size()
    )
    assert benign_attacked.nunique() == 1
    assert int(benign_attacked.iloc[0]) == budget.benign_per_attacked

    benign_only_ids = scenario_df.loc[
        scenario_df["ground_truth_campaign_member"] == 0, "scenario_vehicle_id"
    ].unique()
    benign_only_counts = (
        scenario_df[scenario_df["scenario_vehicle_id"].isin(benign_only_ids)]
        .groupby("scenario_vehicle_id")
        .size()
    )
    assert benign_only_counts.nunique() == 1
    assert int(benign_only_counts.iloc[0]) == budget.benign_per_benign

    assert len(scenario_df) == budget.expected_total_nodes


def build_event_budget_validation_row(
    run_id: str,
    seed: int,
    campaign_size: int,
    scenario_df: pd.DataFrame,
    budget: DescriptorBudget,
) -> dict[str, Any]:
    vehicle_counts = scenario_df.groupby("scenario_vehicle_id").size()
    mal_counts = scenario_df[scenario_df["ground_truth_malicious"] == 1].groupby("scenario_vehicle_id").size()
    ben_counts = scenario_df[scenario_df["ground_truth_malicious"] == 0].groupby("scenario_vehicle_id").size()
    passed = True
    try:
        assert_scenario_budget(scenario_df, budget, campaign_size)
    except AssertionError:
        passed = False
    return {
        "run_id": run_id,
        "seed": seed,
        "campaign_size": campaign_size,
        "expected_total_nodes": budget.expected_total_nodes,
        "actual_total_nodes": len(scenario_df),
        "descriptors_per_vehicle_min": int(vehicle_counts.min()),
        "descriptors_per_vehicle_max": int(vehicle_counts.max()),
        "malicious_descriptors_per_attacked_vehicle_min": int(mal_counts.min()) if not mal_counts.empty else 0,
        "malicious_descriptors_per_attacked_vehicle_max": int(mal_counts.max()) if not mal_counts.empty else 0,
        "benign_descriptors_per_vehicle_min": int(ben_counts.min()) if not ben_counts.empty else 0,
        "benign_descriptors_per_vehicle_max": int(ben_counts.max()) if not ben_counts.empty else 0,
        "validation_passed": passed,
    }


def build_vehicle_composition_row(
    run_id: str,
    seed: int,
    attack_strength: str,
    campaign_size: int,
    scenario_df: pd.DataFrame,
    budget: DescriptorBudget,
    comp: dict[str, int],
) -> dict[str, Any]:
    attacked = scenario_df[scenario_df["ground_truth_campaign_member"] == 1]
    counts = attacked.groupby("vehicle_model").size()
    inst_counts = attacked.groupby("vehicle_model")["scenario_vehicle_id"].nunique().to_dict()
    return {
        "run_id": run_id,
        "seed": seed,
        "attack_strength": attack_strength,
        "campaign_size": campaign_size,
        "total_fleet_size": budget.total_fleet_size,
        "Hyundai_instances": int(inst_counts.get("Hyundai", 0)),
        "Kia_instances": int(inst_counts.get("Kia", 0)),
        "Chevrolet_instances": int(inst_counts.get("Chevrolet", 0)),
        "vehicle_model_diversity": int(attacked["vehicle_model"].nunique()),
        "composition_string": composition_string(comp),
        "Hyundai_descriptors": int(counts.get("Hyundai", 0)),
        "Kia_descriptors": int(counts.get("Kia", 0)),
        "Chevrolet_descriptors": int(counts.get("Chevrolet", 0)),
    }


def generate_corrected_campaign_scenario(
    *,
    attack_strength: AttackStrength,
    seed: int,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    campaign_size: int,
    coordination_strength: float,
    budget: DescriptorBudget | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int], DescriptorBudget]:
    """Build a fixed-budget scenario; raises before method execution if invalid."""
    budget = budget or DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
    )
    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    n_benign = budget.total_fleet_size - campaign_size
    comp = platform_composition(attack_strength, campaign_size, seed)

    attacked_inst, benign_inst = select_instances_for_fleet(
        catalog,
        n_attacked=campaign_size,
        n_benign=n_benign,
        attack_strength=attack_strength,
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

    rng = np.random.default_rng(seed)
    campaign_id = f"CAMP-CORRECTED-{attack_strength.upper()}"
    rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, Any]] = []

    for inst in attacked_inst:
        seg_df = _segment_rows(descriptors, manifest, inst)
        atk_type = _attack_type_for_instance(inst, attack_strength)
        chunk = _build_attacked_vehicle_chunk(
            seg_df,
            attack_strength=attack_strength,
            atk_type=atk_type,
            budget=budget,
            weak_th=weak_th,
            strong_th=strong_th,
            rng=rng,
        )
        _append_vehicle_chunk(
            rows,
            mapping_rows,
            chunk,
            inst,
            "coordinated",
            True,
            campaign_id,
            campaign_size,
            attack_strength,
            seed,
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
        raise ValueError(
            f"Node count {len(scenario_df)} != expected {budget.expected_total_nodes}"
        )

    assert_scenario_budget(scenario_df, budget, campaign_size)

    if coordination_strength > 0:
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

    membership = _build_membership(scenario_df, seed, campaign_size, coordination_strength, attack_strength)
    if (scenario_df["scenario_role"] == "coordinated").any():
        sim = measure_mean_pairwise_similarity(
            scenario_df, scenario_df["scenario_role"] == "coordinated"
        )
        membership["mean_malicious_pairwise_similarity"] = sim

    scenario_df["experiment"] = "campaign_size_corrected"
    scenario_df["attack_strength"] = attack_strength
    scenario_df["configured_campaign_size"] = campaign_size
    scenario_df["configured_coordination_strength"] = coordination_strength
    scenario_df["total_fleet_size"] = budget.total_fleet_size
    scenario_df["descriptors_per_vehicle"] = budget.descriptors_per_vehicle
    scenario_df["model_diversity"] = len({i["vehicle_model"] for i in attacked_inst})

    return scenario_df, mapping_df, membership, comp, budget


def _append_vehicle_chunk(
    rows, mapping_rows, chunk, inst, role, is_attacked, campaign_id,
    campaign_size, attack_strength, seed,
) -> None:
    chunk = chunk.copy()
    chunk["scenario_vehicle_id"] = inst["scenario_vehicle_id"]
    chunk["vehicle_token"] = inst["scenario_vehicle_id"]
    chunk["source_trace"] = source_trace_name(inst["source_file"])
    chunk["source_dataset"] = "Car-Hacking Dataset"
    chunk["scenario_role"] = role
    chunk["ground_truth_campaign_id"] = campaign_id if is_attacked else ""
    chunk["ground_truth_campaign_member"] = int(is_attacked)
    if "ground_truth_malicious" not in chunk.columns:
        chunk["ground_truth_malicious"] = chunk["attack_type"].map(
            lambda a: int(not is_benign_attack_type(a))
        )
    chunk["scenario_gt_malicious"] = chunk["ground_truth_malicious"]
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
            "experiment": "campaign_size_corrected",
            "seed": seed,
        }
    )


def _build_membership(
    scenario_df: pd.DataFrame,
    seed: int,
    campaign_size: int,
    coordination_strength: float,
    attack_strength: str,
) -> pd.DataFrame:
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
                "experiment": "campaign_size_corrected",
                "scenario_id": "CA-campaign_size_corrected",
                "scenario_role": r["scenario_role"],
                "ground_truth_campaign_id": r.get("ground_truth_campaign_id", ""),
                "ground_truth_campaign_member": int(r.get("ground_truth_campaign_member", 0)),
                "ground_truth_malicious": int(r.get("ground_truth_malicious", 0)),
            }
        )
    return pd.DataFrame(membership)
