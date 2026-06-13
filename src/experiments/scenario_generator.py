"""Controlled scenario record selection (S0–S4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.campaign_detection_experiment import CAMPAIGN_ATTACK_TYPES
from src.experiments.coordination_strength import (
    apply_coordination_strength,
    compute_campaign_prototype,
    measure_mean_pairwise_similarity,
)
from src.experiments.data_splits import build_split_manifest, is_benign_attack_type
from src.experiments.scenario_registry import ScenarioSpec
from src.experiments.vehicle_identity import (
    VehicleTokenAllocator,
    assign_identity_to_chunk,
    build_offline_identity_mapping,
)
from src.experiments.vehicle_instance_builder import segment_trace_windows
from src.graph.fleet_graph_builder import event_id_to_window_id

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")
S2_ATTACK_ROTATION = ("flooding", "replay", "fuzzy", "malfunction")


def load_descriptor_tables(
    descriptors_path: Path,
    features_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    descriptors = pd.read_csv(descriptors_path)
    if "window_id" not in descriptors.columns:
        descriptors["window_id"] = descriptors["event_id"].map(event_id_to_window_id)
    features = pd.read_csv(
        features_path,
        usecols=lambda c: c in {
            "window_id", "vehicle_model", "source_file", "attack_type", "label"
        },
    )
    return descriptors, features


def ensure_split_manifest(
    descriptors: pd.DataFrame,
    features: pd.DataFrame,
    *,
    output_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    meta = features.drop_duplicates(subset=["window_id", "vehicle_model", "source_file"])
    manifest = build_split_manifest(
        meta,
        train_ratio=train_ratio,
        validation_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    manifest.to_csv(output_path, index=False)
    return manifest


def _sample_rows(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))).copy()


def _benign_pool(descriptors: pd.DataFrame, split: str, manifest: pd.DataFrame) -> pd.DataFrame:
    desc = _attach_split(descriptors, manifest)
    m = (desc["split"] == split) & desc["attack_type"].map(is_benign_attack_type)
    return desc.loc[m]


def _attach_split(descriptors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns and c in manifest.columns]
    if not join_cols:
        return descriptors
    return descriptors.merge(
        manifest[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )


def _attack_pool(
    descriptors: pd.DataFrame,
    split: str,
    manifest: pd.DataFrame,
    *,
    attack_type: str,
    evidence: str | None = None,
) -> pd.DataFrame:
    desc = _attach_split(descriptors, manifest)
    m = (desc["split"] == split) & (desc["attack_type"] == attack_type)
    if evidence == "strong":
        m &= desc["evidence_level"] == "strong_local_anomaly"
    elif evidence == "weak":
        m &= desc["evidence_level"] == "weak_suspicious_signal"
    return desc.loc[m]


def _attack_segment_instances(
    descriptors: pd.DataFrame,
    split: str,
    manifest: pd.DataFrame,
    *,
    min_windows: int = 10,
    min_attack_events: int = 3,
) -> list[dict[str, Any]]:
    """Distinct test-split attack instances from non-overlapping window segments."""
    desc = _attach_split(descriptors, manifest)
    mal = ~desc["attack_type"].map(is_benign_attack_type)
    desc = desc.loc[(desc["split"] == split) & mal]
    instances: list[dict[str, Any]] = []
    group_cols = [c for c in ("vehicle_model", "source_file") if c in desc.columns]
    if not group_cols:
        return instances
    for key, grp in desc.groupby(group_cols, sort=True):
        vm = key[0] if isinstance(key, tuple) else key
        sf = key[1] if isinstance(key, tuple) and len(key) > 1 else ""
        wins = sorted(grp["window_id"].astype(int).tolist())
        for seg_start, seg_end in segment_trace_windows(wins, min_windows=min_windows):
            seg = grp[(grp["window_id"] >= seg_start) & (grp["window_id"] <= seg_end)]
            if len(seg) < min_attack_events:
                continue
            instances.append(
                {
                    "vehicle_model": vm,
                    "source_file": sf,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "pool": seg,
                }
            )
    return instances


def _select_disjoint_instances(
    candidates: list[dict[str, Any]],
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Pick n instances with non-overlapping window_id sets."""
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    chosen: list[dict[str, Any]] = []
    used_windows: set[int] = set()
    for inst in shuffled:
        wids = set(inst["pool"]["window_id"].astype(int).tolist())
        if used_windows & wids:
            continue
        chosen.append(inst)
        used_windows |= wids
        if len(chosen) >= n:
            break
    return chosen


def _sample_attack_instance(
    instance: dict[str, Any],
    *,
    attack_type: str,
    split: str,
    manifest: pd.DataFrame,
    descriptors: pd.DataFrame,
    config: dict[str, Any],
    evidence: str = "strong",
    max_events: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sample attack rows for one trace instance with evidence and attack-type fallbacks."""
    base = instance["pool"]
    attack_candidates: list[str | None] = []
    if attack_type:
        attack_candidates.append(attack_type)
    for atk in sorted(base["attack_type"].astype(str).unique()):
        if atk not in attack_candidates:
            attack_candidates.append(atk)

    for atk in attack_candidates:
        pool = base if not atk else base[base["attack_type"] == atk]
        if pool.empty:
            continue
        if evidence == "strong":
            filtered = pool[pool["evidence_level"] == "strong_local_anomaly"]
            if not filtered.empty:
                pool = filtered
        elif evidence == "weak":
            filtered = pool[pool["evidence_level"] == "weak_suspicious_signal"]
            if not filtered.empty:
                pool = filtered
        if not pool.empty:
            return _sample_rows(pool, max_events, rng)

        pool = _resolve_coordinated_pool(
            descriptors,
            split,
            manifest,
            vehicle=str(instance["vehicle_model"]),
            attack_type=str(atk or attack_type),
            evidence=evidence if evidence != "none" else "strong",
            config=config,
        )
        if "source_file" in pool.columns and instance.get("source_file"):
            pool = pool[pool["source_file"].astype(str) == str(instance["source_file"])]
        if not pool.empty:
            return _sample_rows(pool, max_events, rng)

    return base.head(0)


def _resolve_coordinated_pool(
    descriptors: pd.DataFrame,
    split: str,
    manifest: pd.DataFrame,
    *,
    vehicle: str,
    attack_type: str,
    evidence: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select attack windows with evidence-aware fallbacks for test-split scarcity."""
    pool = _attack_pool(descriptors, split, manifest, attack_type=attack_type, evidence=evidence)
    pool = pool[pool["vehicle_model"] == vehicle]
    if not pool.empty:
        return pool

    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    pool = _attack_pool(descriptors, split, manifest, attack_type=attack_type, evidence=None)
    pool = pool[pool["vehicle_model"] == vehicle]
    if evidence == "weak":
        pool = pool[(pool["anomaly_score"] >= weak_th) & (pool["anomaly_score"] < strong_th)]
    elif evidence == "strong":
        pool = pool[pool["anomaly_score"] >= strong_th]
    if not pool.empty:
        return pool

    for alt in CAMPAIGN_ATTACK_TYPES:
        if alt == attack_type:
            continue
        pool = _attack_pool(descriptors, split, manifest, attack_type=alt, evidence=evidence)
        pool = pool[pool["vehicle_model"] == vehicle]
        if not pool.empty:
            return pool
    return pool.head(0)


def generate_scenario_records(
    spec: ScenarioSpec,
    *,
    seed: int,
    campaign_size: int,
    coordination_strength: float,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    max_events_per_vehicle: int = 120,
    max_benign_events: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build scenario descriptor table and membership provenance.

    Returns (scenario_descriptors, scenario_membership).
    """
    rng = np.random.default_rng(seed)
    split = "test"
    attack_cfg = config.get("attack_families", {})
    rows: list[pd.DataFrame] = []
    membership: list[dict[str, Any]] = []
    allocator = VehicleTokenAllocator()
    instance_counter = 0

    def _add_chunk(
        chunk: pd.DataFrame,
        role: str,
        campaign_id: str | None,
        *,
        vehicle_model: str | None = None,
        instance_key: str | None = None,
    ) -> None:
        nonlocal instance_counter
        if chunk.empty:
            return
        instance_counter += 1
        vm = vehicle_model or str(chunk["vehicle_model"].iloc[0])
        sf = str(chunk["source_file"].iloc[0]) if "source_file" in chunk.columns else ""
        key = instance_key or f"{spec.key}-{seed}-{instance_counter}"
        wmin = int(chunk["window_id"].min()) if "window_id" in chunk.columns else None
        wmax = int(chunk["window_id"].max()) if "window_id" in chunk.columns else None
        chunk = assign_identity_to_chunk(
            chunk,
            allocator=allocator,
            instance_key=key,
            vehicle_model=vm,
            source_file=sf,
            segment_start=wmin,
            segment_end=wmax,
        )
        chunk["scenario_role"] = role
        chunk["ground_truth_campaign_id"] = campaign_id or ""
        chunk["ground_truth_campaign_member"] = int(role in {"attacked", "coordinated"})
        chunk["scenario_gt_malicious"] = int(role in {"attacked", "coordinated"})
        chunk["ground_truth_malicious"] = int(role in {"attacked", "coordinated"})
        rows.append(chunk)
        for _, r in chunk.iterrows():
            membership.append(
                {
                    "event_id": r["event_id"],
                    "window_id": int(r["window_id"]),
                    "vehicle_token": r["vehicle_token"],
                    "scenario_vehicle_id": r["scenario_vehicle_id"],
                    "vehicle_model": r["vehicle_model"],
                    "source_file": r.get("source_file", ""),
                    "source_trace": r.get("source_trace", ""),
                    "attack_type": r["attack_type"],
                    "split": split,
                    "scenario": spec.key,
                    "scenario_id": spec.scenario_id,
                    "seed": seed,
                    "campaign_size": campaign_size,
                    "coordination_strength": coordination_strength,
                    "scenario_role": role,
                    "ground_truth_campaign_id": campaign_id or "",
                    "ground_truth_campaign_member": int(role in {"attacked", "coordinated"}),
                    "ground_truth_malicious": int(role in {"attacked", "coordinated"}),
                }
            )

    if spec.scenario_id == "S0":
        pool = _benign_pool(descriptors, split, manifest)
        chunk = _sample_rows(pool, max_benign_events, rng)
        _add_chunk(chunk, "benign", None)

    elif spec.scenario_id == "S1":
        attack = attack_cfg.get("strong_default", "flooding")
        vehicles = list(VEHICLE_MODELS)
        rng.shuffle(vehicles)
        atk_chunk = pd.DataFrame()
        attacked_vehicle = vehicles[0]
        for v in vehicles:
            atk_pool = _attack_pool(descriptors, split, manifest, attack_type=attack, evidence="strong")
            atk_pool = atk_pool[atk_pool["vehicle_model"] == v]
            if not atk_pool.empty:
                attacked_vehicle = v
                atk_chunk = _sample_rows(atk_pool, max_events_per_vehicle, rng)
                break
        if atk_chunk.empty:
            for v in vehicles:
                atk_pool = _attack_pool(descriptors, split, manifest, attack_type=attack, evidence=None)
                atk_pool = atk_pool[atk_pool["vehicle_model"] == v]
                if not atk_pool.empty:
                    attacked_vehicle = v
                    atk_chunk = _sample_rows(atk_pool, max_events_per_vehicle, rng)
                    break
        _add_chunk(atk_chunk, "attacked", None, vehicle_model=attacked_vehicle)
        for v in vehicles:
            if v == attacked_vehicle:
                continue
            ben = _benign_pool(descriptors, split, manifest)
            ben = ben[ben["vehicle_model"] == v]
            _add_chunk(
                _sample_rows(ben, max_events_per_vehicle // 2, rng),
                "benign",
                None,
                vehicle_model=v,
            )

    elif spec.scenario_id == "S2":
        n = max(2, campaign_size)
        candidates = _attack_segment_instances(descriptors, split, manifest)
        selected = _select_disjoint_instances(candidates, n, rng)
        if len(selected) < n:
            raise RuntimeError(
                f"S2 campaign_size={n} requires {n} disjoint attack instances; "
                f"only {len(selected)} non-overlapping segments available in test split"
            )
        for i, inst in enumerate(selected):
            attack = S2_ATTACK_ROTATION[i % len(S2_ATTACK_ROTATION)]
            chunk = _sample_attack_instance(
                inst,
                attack_type=attack,
                split=split,
                manifest=manifest,
                descriptors=descriptors,
                config=config,
                evidence="strong",
                max_events=max_events_per_vehicle,
                rng=rng,
            )
            if chunk.empty:
                raise RuntimeError(
                    f"S2 cannot sample attack_type={attack} for instance "
                    f"{inst['vehicle_model']}/{inst.get('source_file', '')}"
                )
            cid = f"INCIDENT-{attack}-{i}"
            _add_chunk(
                chunk,
                "attacked",
                cid,
                vehicle_model=str(inst["vehicle_model"]),
                instance_key=f"{spec.key}-{seed}-s2-{i}-{inst.get('source_file', '')}",
            )
        ben = _benign_pool(descriptors, split, manifest)
        _add_chunk(_sample_rows(ben, max_benign_events, rng), "benign", None)

    elif spec.scenario_id in ("S3", "S4"):
        n = max(2, campaign_size)
        evidence = "strong" if spec.scenario_id == "S3" else "weak"
        attack = attack_cfg.get(
            "strong_default" if spec.scenario_id == "S3" else "weak_default",
            "flooding",
        )
        campaign_id = spec.ground_truth_campaign_label or f"CAMP-{spec.scenario_id}"
        candidates = _attack_segment_instances(descriptors, split, manifest)
        pool_size = min(len(candidates), max(n * 5, n))
        selected = _select_disjoint_instances(candidates, pool_size, rng)
        if len(selected) < n:
            raise RuntimeError(
                f"{spec.scenario_id} campaign_size={n} requires {n} disjoint attack instances; "
                f"only {len(selected)} non-overlapping segments available"
            )
        malicious_parts: list[tuple[pd.DataFrame, str, int, dict[str, Any]]] = []
        for i, inst in enumerate(selected):
            part = _sample_attack_instance(
                inst,
                attack_type=attack,
                split=split,
                manifest=manifest,
                descriptors=descriptors,
                config=config,
                evidence=evidence,
                max_events=max_events_per_vehicle,
                rng=rng,
            )
            if not part.empty:
                malicious_parts.append((part.copy(), str(inst["vehicle_model"]), i, inst))
            if len(malicious_parts) >= n:
                break
        malicious_parts = malicious_parts[:n]
        if len(malicious_parts) < n:
            raise RuntimeError(
                f"{spec.scenario_id} could assemble only {len(malicious_parts)}/{n} coordinated instances "
                f"for attack={attack} evidence={evidence}"
            )
        coordinated_frames: list[pd.DataFrame] = []
        for part, _vehicle, _i, _inst in malicious_parts:
            coordinated_frames.append(part)
        malicious = (
            pd.concat(coordinated_frames, ignore_index=True) if coordinated_frames else pd.DataFrame()
        )

        if not malicious.empty and coordination_strength > 0:
            proto = compute_campaign_prototype(
                descriptors,
                attack_type=attack,
            )
            mask = pd.Series(True, index=malicious.index)
            malicious, _prov = apply_coordination_strength(
                malicious,
                strength=coordination_strength,
                campaign_prototype=proto,
                target_mask=mask,
                seed=seed,
            )

        for part, vehicle, i, inst in malicious_parts:
            part_ids = set(part["event_id"])
            sub = malicious[malicious["event_id"].isin(part_ids)]
            _add_chunk(
                sub,
                "coordinated",
                campaign_id,
                vehicle_model=vehicle,
                instance_key=f"{spec.key}-{seed}-coord-{i}-{inst.get('source_file', '')}",
            )
        ben = _benign_pool(descriptors, split, manifest)
        _add_chunk(_sample_rows(ben, max_benign_events, rng), "benign", None)

    else:
        raise ValueError(f"Unsupported scenario {spec.scenario_id}")

    if not rows:
        raise RuntimeError(f"No records generated for {spec.key} seed={seed}")

    scenario_df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])
    scenario_df["scenario_key"] = spec.key
    scenario_df["scenario_seed"] = seed
    scenario_df["configured_campaign_size"] = campaign_size
    scenario_df["configured_coordination_strength"] = coordination_strength

    # Subsample if too large for graph methods
    max_nodes = int(config.get("graph", {}).get("max_nodes_per_scenario", 1500))
    if len(scenario_df) > max_nodes:
        attacked = scenario_df[scenario_df["scenario_role"] != "benign"]
        benign = scenario_df[scenario_df["scenario_role"] == "benign"]
        n_benign = max(max_nodes - len(attacked), 0)
        benign = _sample_rows(benign, n_benign, rng)
        scenario_df = pd.concat([attacked, benign], ignore_index=True)

    membership_df = pd.DataFrame(membership)
    identity_map = build_offline_identity_mapping(scenario_df)
    scenario_df.attrs["offline_identity_mapping"] = identity_map
    if spec.scenario_id in ("S3", "S4") and not scenario_df.empty:
        mal_mask = scenario_df["scenario_role"] == "coordinated"
        sim = measure_mean_pairwise_similarity(scenario_df, mal_mask)
        membership_df["mean_malicious_pairwise_similarity"] = sim

    return scenario_df, membership_df


def save_scenario_membership(records: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_path, index=False)
    return output_path
