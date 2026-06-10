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
    local_cfg = config.get("local_ids", {})
    attack_cfg = config.get("attack_families", {})
    rows: list[pd.DataFrame] = []
    membership: list[dict[str, Any]] = []

    def _add_chunk(chunk: pd.DataFrame, role: str, campaign_id: str | None) -> None:
        if chunk.empty:
            return
        chunk = chunk.copy()
        chunk["scenario_role"] = role
        chunk["ground_truth_campaign_id"] = campaign_id
        chunk["scenario_gt_malicious"] = int(role in {"attacked", "coordinated"})
        rows.append(chunk)
        for _, r in chunk.iterrows():
            membership.append(
                {
                    "event_id": r["event_id"],
                    "window_id": int(r["window_id"]),
                    "vehicle_model": r["vehicle_model"],
                    "source_file": r.get("source_file", ""),
                    "attack_type": r["attack_type"],
                    "split": split,
                    "scenario": spec.key,
                    "scenario_id": spec.scenario_id,
                    "seed": seed,
                    "campaign_size": campaign_size,
                    "coordination_strength": coordination_strength,
                    "scenario_role": role,
                    "ground_truth_campaign_id": campaign_id or "",
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
        _add_chunk(atk_chunk, "attacked", None)
        for v in vehicles:
            if v == attacked_vehicle:
                continue
            ben = _benign_pool(descriptors, split, manifest)
            ben = ben[ben["vehicle_model"] == v]
            _add_chunk(_sample_rows(ben, max_events_per_vehicle // 2, rng), "benign", None)

    elif spec.scenario_id == "S2":
        n = max(2, campaign_size)
        vehicles = list(VEHICLE_MODELS)
        while len(vehicles) < n:
            vehicles.extend(VEHICLE_MODELS)
        vehicles = vehicles[:n]
        for i, vehicle in enumerate(vehicles):
            attack = S2_ATTACK_ROTATION[i % len(S2_ATTACK_ROTATION)]
            pool = _attack_pool(descriptors, split, manifest, attack_type=attack, evidence="strong")
            pool = pool[pool["vehicle_model"] == vehicle]
            cid = f"INCIDENT-{attack}-{vehicle}"
            _add_chunk(_sample_rows(pool, max_events_per_vehicle, rng), "attacked", cid)
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
        vehicles = list(VEHICLE_MODELS)
        while len(vehicles) < n:
            vehicles.extend(VEHICLE_MODELS)
        vehicles = vehicles[:n]
        malicious_parts: list[pd.DataFrame] = []
        for vehicle in vehicles:
            pool = _resolve_coordinated_pool(
                descriptors,
                split,
                manifest,
                vehicle=vehicle,
                attack_type=attack,
                evidence=evidence,
                config=config,
            )
            malicious_parts.append(_sample_rows(pool, max_events_per_vehicle, rng))
        malicious = pd.concat(malicious_parts, ignore_index=True) if malicious_parts else pd.DataFrame()

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
        _add_chunk(malicious, "coordinated", campaign_id)
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
    if spec.scenario_id in ("S3", "S4") and not scenario_df.empty:
        mal_mask = scenario_df["scenario_role"] == "coordinated"
        sim = measure_mean_pairwise_similarity(scenario_df, mal_mask)
        membership_df["mean_malicious_pairwise_similarity"] = sim

    return scenario_df, membership_df


def save_scenario_membership(records: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_path, index=False)
    return output_path
