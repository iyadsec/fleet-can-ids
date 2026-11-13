"""Construct disjoint scenario vehicle instances from held-out test traces/segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.experiments.data_splits import is_benign_attack_type

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")
MODEL_DISPLAY = {
    "Hyundai": "Hyundai Sonata",
    "Kia": "Kia Soul",
    "Chevrolet": "Chevrolet Spark",
}

AttackStrength = Literal["strong", "weak"]


def source_trace_name(source_file: str) -> str:
    return Path(str(source_file)).name


def _contiguous_blocks(window_ids: list[int]) -> list[list[int]]:
    if not window_ids:
        return []
    wins = sorted(window_ids)
    blocks: list[list[int]] = []
    block = [wins[0]]
    for w in wins[1:]:
        if w == block[-1] + 1:
            block.append(w)
        else:
            blocks.append(block)
            block = [w]
    blocks.append(block)
    return blocks


def segment_trace_windows(
    window_ids: list[int],
    *,
    min_windows: int,
) -> list[tuple[int, int]]:
    """
    Partition sorted windows into disjoint sequential chunks of min_windows.

    Chunks do not share window_ids (required for independent instances). When
    window_ids are sparse in time, segment_start/segment_end are the min/max
    window_id in each chunk (ranges may span gaps; overlap is validated on IDs).
    """
    wins = sorted(window_ids)
    segments: list[tuple[int, int]] = []
    for start in range(0, len(wins), min_windows):
        chunk = wins[start : start + min_windows]
        if len(chunk) >= min_windows:
            segments.append((chunk[0], chunk[-1]))
    return segments


def _classify_event_strength(row: pd.Series, weak_th: float, strong_th: float) -> str:
    if is_benign_attack_type(row.get("attack_type", "")):
        return "benign"
    score = float(row.get("anomaly_score", 0.0))
    if score >= strong_th:
        return "strong"
    if score >= weak_th:
        return "weak"
    return "other"


def build_instance_catalog(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    weak_threshold: float = 0.55,
    strong_threshold: float = 0.80,
    min_windows_per_segment: int = 15,
    min_strong_events: int = 3,
    min_weak_events: int = 3,
    min_benign_events: int = 5,
    target_split: str = "test",
) -> pd.DataFrame:
    """Catalog disjoint trace segments eligible as scenario vehicle instances."""
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns]
    desc = descriptors.drop(columns=["split"], errors="ignore").merge(
        manifest[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )
    test = desc[desc["split"] == target_split].copy()
    rows: list[dict[str, Any]] = []
    instance_counter = 0

    for (vehicle_model, source_file), grp in test.groupby(["vehicle_model", "source_file"], sort=True):
        trace = source_trace_name(source_file)
        wins = sorted(grp["window_id"].astype(int).tolist())
        for start in range(0, len(wins), min_windows_per_segment):
            chunk_ids = wins[start : start + min_windows_per_segment]
            if len(chunk_ids) < min_windows_per_segment:
                continue
            seg_start, seg_end = chunk_ids[0], chunk_ids[-1]
            seg = grp[grp["window_id"].isin(chunk_ids)]
            strengths = seg.apply(
                lambda r: _classify_event_strength(r, weak_threshold, strong_threshold),
                axis=1,
            )
            strong_n = int((strengths == "strong").sum())
            weak_n = int((strengths == "weak").sum())
            benign_n = int((strengths == "benign").sum())
            attack_types = sorted(
                {
                    str(a)
                    for a in seg.loc[~seg["attack_type"].map(is_benign_attack_type), "attack_type"].unique()
                }
            )
            instance_counter += 1
            sid = f"V_{instance_counter:04d}"
            eligible_campaign = strong_n >= min_strong_events or weak_n >= min_weak_events
            eligible_diversity = eligible_campaign
            rows.append(
                {
                    "scenario_vehicle_id": sid,
                    "vehicle_model": vehicle_model,
                    "source_trace": trace,
                    "source_file": str(source_file),
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "window_ids": chunk_ids,
                    "split": target_split,
                    "attack_types_available": "|".join(attack_types) if attack_types else "benign",
                    "strong_attack_events": strong_n,
                    "weak_attack_events": weak_n,
                    "benign_events": benign_n,
                    "eligible_for_campaign_size": eligible_campaign,
                    "eligible_for_model_diversity": eligible_diversity,
                }
            )
    return pd.DataFrame(rows)


def _segment_rows(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    instance: dict[str, Any],
) -> pd.DataFrame:
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns]
    desc = descriptors.drop(columns=["split"], errors="ignore").merge(
        manifest[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )
    target_split = str(instance.get("split", "test"))
    m = (
        (desc["split"] == target_split)
        & (desc["vehicle_model"] == instance["vehicle_model"])
        & (desc["source_file"].astype(str) == str(instance["source_file"]))
    )
    seg = desc.loc[m].copy()
    if "window_ids" in instance and instance["window_ids"]:
        allowed = {int(w) for w in instance["window_ids"]}
        seg = seg[seg["window_id"].isin(allowed)]
    else:
        seg = seg[
            (seg["window_id"] >= instance["segment_start"])
            & (seg["window_id"] <= instance["segment_end"])
        ]
    return seg


def _filter_attack_strength(
    df: pd.DataFrame,
    attack_strength: AttackStrength,
    attack_type: str,
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:
    mal = df[~df["attack_type"].map(is_benign_attack_type)].copy()
    if attack_type:
        typed = mal[mal["attack_type"] == attack_type]
        if not typed.empty:
            mal = typed
    if attack_strength == "strong":
        return mal[mal["anomaly_score"] >= strong_threshold]
    return mal[(mal["anomaly_score"] >= weak_threshold) & (mal["anomaly_score"] < strong_threshold)]


def select_instances_for_fleet(
    catalog: pd.DataFrame,
    *,
    n_attacked: int,
    n_benign: int,
    attack_strength: AttackStrength,
    attack_type: str,
    model_composition: dict[str, int] | None,
    seed: int,
    weak_threshold: float = 0.55,
    strong_threshold: float = 0.80,
    min_attack_events: int = 3,
    min_benign_on_attacked: int = 0,
    min_benign_on_benign: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select disjoint attacked and benign instances deterministically."""
    rng = np.random.default_rng(seed)
    catalog = catalog.copy()
    atk_col = "strong_attack_events" if attack_strength == "strong" else "weak_attack_events"

    if model_composition:
        attacked: list[dict[str, Any]] = []
        for model, count in model_composition.items():
            if count <= 0:
                continue
            pool = catalog[
                (catalog["vehicle_model"] == model)
                & catalog["eligible_for_model_diversity"]
                & (catalog[atk_col] >= min_attack_events)
            ]
            if min_benign_on_attacked > 0:
                pool = pool[pool["benign_events"] >= min_benign_on_attacked]
            if pool.empty:
                raise ValueError(
                    f"Insufficient {attack_strength} instances for {model} "
                    f"(need {count}, pool={len(pool)})"
                )
            chosen = pool.sample(n=min(count, len(pool)), random_state=int(rng.integers(0, 2**31 - 1)))
            attacked.extend(chosen.to_dict("records"))
        if len(attacked) < n_attacked:
            raise ValueError(
                f"Model composition supplied {len(attacked)} attacked instances; need {n_attacked}"
            )
        attacked = attacked[:n_attacked]
    else:
        pool = catalog[
            catalog["eligible_for_campaign_size"]
            & (catalog[atk_col] >= min_attack_events)
        ]
        if min_benign_on_attacked > 0:
            pool = pool[pool["benign_events"] >= min_benign_on_attacked]
        if len(pool) < n_attacked:
            raise ValueError(
                f"Insufficient {attack_strength} attacked instances: need {n_attacked}, have {len(pool)}"
            )
        attacked_df = pool.sample(n=n_attacked, random_state=int(rng.integers(0, 2**31 - 1)))
        attacked = attacked_df.to_dict("records")

    used_ids = {a["scenario_vehicle_id"] for a in attacked}
    ben_pool = catalog[
        catalog["attack_types_available"].eq("benign")
        & (catalog["benign_events"] >= min_benign_on_benign)
        & ~catalog["scenario_vehicle_id"].isin(used_ids)
    ]
    if len(ben_pool) < n_benign:
        raise ValueError(f"Insufficient benign instances: need {n_benign}, have {len(ben_pool)}")
    benign_df = ben_pool.sample(n=n_benign, random_state=int(rng.integers(0, 2**31 - 1)))
    benign = benign_df.to_dict("records")
    return attacked, benign


def select_instances_with_benign_composition(
    catalog: pd.DataFrame,
    *,
    n_attacked: int,
    benign_model_composition: dict[str, int],
    attack_strength: AttackStrength,
    attack_type: str,
    model_composition: dict[str, int] | None,
    seed: int,
    weak_threshold: float = 0.55,
    strong_threshold: float = 0.80,
    min_attack_events: int = 3,
    min_benign_on_benign: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select attacked instances plus a fixed per-model benign fleet composition."""
    attacked, _ = select_instances_for_fleet(
        catalog,
        n_attacked=n_attacked,
        n_benign=0,
        attack_strength=attack_strength,
        attack_type=attack_type,
        model_composition=model_composition,
        seed=seed,
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
        min_attack_events=min_attack_events,
        min_benign_on_attacked=0,
        min_benign_on_benign=min_benign_on_benign,
    )
    rng = np.random.default_rng(seed)
    used_ids = {a["scenario_vehicle_id"] for a in attacked}
    benign: list[dict[str, Any]] = []
    for model, count in benign_model_composition.items():
        if count <= 0:
            continue
        pool = catalog[
            (catalog["vehicle_model"] == model)
            & catalog["attack_types_available"].eq("benign")
            & (catalog["benign_events"] >= min_benign_on_benign)
            & ~catalog["scenario_vehicle_id"].isin(used_ids)
        ]
        if len(pool) < count:
            raise ValueError(
                f"Insufficient benign {model} instances: need {count}, pool={len(pool)}"
            )
        chosen = pool.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
        benign.extend(chosen.to_dict("records"))
        used_ids.update(chosen["scenario_vehicle_id"].tolist())
    expected = sum(benign_model_composition.values())
    if len(benign) != expected:
        raise ValueError(f"Benign composition mismatch: got {len(benign)}, expected {expected}")
    return attacked, benign


def validate_instance_selection(
    attacked: list[dict[str, Any]],
    benign: list[dict[str, Any]],
    *,
    configured_campaign_size: int,
    configured_model_composition: dict[str, int] | None = None,
) -> list[str]:
    """Return validation errors (empty if valid)."""
    errors: list[str] = []
    all_inst = attacked + benign
    ids = [i["scenario_vehicle_id"] for i in all_inst]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate scenario_vehicle_id in fleet selection")

    ranges: list[tuple[str, str, int, int]] = []
    for inst in all_inst:
        ranges.append(
            (
                str(inst["source_file"]),
                inst["scenario_vehicle_id"],
                int(inst["segment_start"]),
                int(inst["segment_end"]),
            )
        )
    by_file: dict[str, list[tuple[str, int, int]]] = {}
    for sf, sid, a, b in ranges:
        by_file.setdefault(sf, []).append((sid, a, b))
    window_sets: list[tuple[str, set[int]]] = []
    for inst in all_inst:
        wids = inst.get("window_ids") or []
        window_sets.append((inst["scenario_vehicle_id"], {int(w) for w in wids}))
    for i in range(len(window_sets)):
        for j in range(i + 1, len(window_sets)):
            a_id, a_set = window_sets[i]
            b_id, b_set = window_sets[j]
            overlap = a_set & b_set
            if overlap:
                errors.append(
                    f"Overlapping window_ids between {a_id} and {b_id}: {len(overlap)} shared"
                )

    if len(attacked) != configured_campaign_size:
        errors.append(
            f"Campaign size mismatch: configured={configured_campaign_size}, "
            f"attacked_instances={len(attacked)}"
        )

    if configured_model_composition:
        actual = pd.Series([a["vehicle_model"] for a in attacked]).value_counts().to_dict()
        for model, expected in configured_model_composition.items():
            if expected and actual.get(model, 0) != expected:
                errors.append(
                    f"Model composition mismatch for {model}: expected {expected}, got {actual.get(model, 0)}"
                )

    return errors


def validate_scenario_records(
    scenario_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    *,
    configured_campaign_size: int,
) -> list[str]:
    """Validate built scenario has no duplicate events and correct campaign membership."""
    errors: list[str] = []
    if scenario_df["event_id"].duplicated().any():
        errors.append("Duplicated event_id in scenario records")
    attacked_ids = scenario_df.loc[
        scenario_df["ground_truth_campaign_member"] == 1, "scenario_vehicle_id"
    ].unique()
    if len(attacked_ids) != configured_campaign_size:
        errors.append(
            f"Attacked scenario_vehicle_id count {len(attacked_ids)} != {configured_campaign_size}"
        )
    for _, row in mapping_df.iterrows():
        wids = row.get("window_ids")
        if wids is None or (isinstance(wids, float) and pd.isna(wids)):
            continue
        if isinstance(wids, str) and wids:
            wids = [int(x) for x in wids.split("|") if x]
        allowed = {int(w) for w in wids}
        seg = scenario_df[scenario_df["scenario_vehicle_id"] == row["scenario_vehicle_id"]]
        if seg.empty:
            continue
        used = set(seg["window_id"].astype(int).tolist())
        if allowed and not used.issubset(allowed):
            errors.append(f"Window leakage outside partition for {row['scenario_vehicle_id']}")
    return errors
