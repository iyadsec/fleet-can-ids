#!/usr/bin/env python3
"""Publication-faithful scenario construction for corrected ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fleet_scaler import FEATURE_NAMES
from scenario_builder_corrected import (
    BEHAVIOURAL_24,
    apply_coordination_strength,
    compute_campaign_prototype,
    freeze_scenario,
    load_frozen_scenario,
    pack_frozen,
    platform_composition,
    _derive_gnn9,
)


def _sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Need {n} rows, have {len(df)}")
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].copy()


def _preferred_attack(strength: str, vehicle_model: str) -> str:
    if strength == "strong" and vehicle_model == "Chevrolet":
        return "fuzzy"
    return "malfunction"


def _band_mask(
    df: pd.DataFrame, strength: str, weak_thr: float, strong_thr: float
) -> pd.Series:
    mal = df["label"].astype(int) == 1
    s = df["anomaly_score"].astype(float)
    if strength == "strong":
        return mal & (s >= strong_thr)
    return mal & (s >= weak_thr) & (s < strong_thr)


def _event(
    src: pd.Series,
    vehicle_id: str,
    window_id: int,
    is_attack: bool,
    campaign_id: int,
    attack_family: str,
    strong_thr: float,
    weak_thr: float,
    scenario: str,
    seed: int,
    *,
    coordinated: bool,
) -> dict[str, Any]:
    score = float(src["anomaly_score"])
    local_alert = int(score >= strong_thr)
    weak_signal = int((not local_alert) and score >= weak_thr)
    if is_attack and not local_alert and not weak_signal:
        weak_signal = 1
    evidence = "strong_local_anomaly" if local_alert else "weak_suspicious_signal"
    gt_cid = int(campaign_id) if coordinated else (-1 if not is_attack else int(campaign_id))
    if coordinated:
        gt_cid = 0
    elif not is_attack:
        gt_cid = -1
    row: dict[str, Any] = {
        "event_id": f"{vehicle_id}-w{window_id:02d}",
        "vehicle_id": vehicle_id,
        "window_id": int(window_id),
        "is_attack": int(is_attack),
        "gt_campaign_id": gt_cid,
        "attack_family": attack_family,
        "anomaly_score": score,
        "evidence_level": evidence,
        "local_alert": local_alert,
        "weak_signal": weak_signal,
        "scenario": scenario,
        "seed": int(seed),
        "synthetic": 0,
        "source_window_uid": str(src["window_uid"]),
        "physical_vehicle": src.get("physical_vehicle", src.get("vehicle_model")),
        "oem_platform": src.get("vehicle_model"),
        "source_trace": src.get("source_trace", src.get("source_basename")),
        # Fleet instance id for campaign-gate vehicle support (not OEM name).
        "vehicle_model": vehicle_id,
        "attack_type": src.get("attack_type"),
        "scenario_role": "coordinated" if coordinated else "background",
        "ground_truth_campaign_member": int(coordinated),
        "ground_truth_malicious": int(is_attack),
    }
    for c in BEHAVIOURAL_24:
        if c in src.index:
            row[c] = float(src[c]) if pd.notna(src[c]) else np.nan
    for c in FEATURE_NAMES:
        if c in src.index and c not in row:
            row[c] = float(src[c]) if pd.notna(src[c]) else np.nan
    return row


def build_scenario_from_pool(
    scenario: str,
    seed: int,
    pool: pd.DataFrame,
    *,
    n_vehicles: int = 20,
    windows_per_vehicle: int = 10,
    campaign_size: int = 5,
    strong_thr: float = 0.80,
    weak_thr: float = 0.55,
    coordination_strength: float = 1.0,
    prototype_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    notes: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "coordination_strength": coordination_strength,
        "method": "prototype_blend_with_bounded_noise"
        if scenario.endswith("_campaign")
        else "none",
        "sampling_adjustments": [],
    }
    proto_src = prototype_pool if prototype_pool is not None else pool
    used: set[str] = set()

    def take(df: pd.DataFrame, n: int) -> pd.DataFrame:
        cand = df[~df["window_uid"].astype(str).isin(used)]
        if len(cand) < n:
            raise ValueError(f"Insufficient unused windows: need {n}, have {len(cand)}")
        chosen = _sample(cand, n, rng)
        used.update(chosen["window_uid"].astype(str).tolist())
        return chosen

    rows: list[dict[str, Any]] = []

    if scenario in ("strong_campaign", "weak_campaign"):
        strength = "strong" if scenario == "strong_campaign" else "weak"
        comp = platform_composition(strength, campaign_size)
        notes["platform_composition"] = comp
        band = pool[_band_mask(pool, strength, weak_thr, strong_thr)]
        if len(band) < campaign_size * 5:
            raise RuntimeError(
                f"Insufficient {strength} malicious TEST windows in score band: "
                f"have {len(band)}, need >={campaign_size * 5} "
                f"(weak_thr={weak_thr}, strong_thr={strong_thr})."
            )

        attacked_vms: list[str] = []
        for vm, cnt in comp.items():
            attacked_vms.extend([vm] * int(cnt))

        primary_attack = "malfunction"
        fam_counts = band["attack_type"].value_counts()
        if primary_attack not in fam_counts.index:
            primary_attack = str(fam_counts.index[0])
            notes["sampling_adjustments"].append(
                f"primary_attack fallback→{primary_attack} (no malfunction in band)"
            )
        notes["primary_attack"] = primary_attack

        for i, vm in enumerate(attacked_vms):
            vid = f"atk_{vm}_{i}"
            pref = _preferred_attack(strength, vm)
            fam_pool = band[(band["vehicle_model"] == vm) & (band["attack_type"] == pref)]
            if len(fam_pool) < 5:
                fam_pool = band[band["vehicle_model"] == vm]
            if len(fam_pool) < 5:
                fam_pool = band
                notes["sampling_adjustments"].append(
                    f"{vid}: expanded to all-platform {strength} band"
                )
            mal = take(fam_pool, 5)
            ben_pool = pool[(pool["label"] == 0) & (pool["vehicle_model"] == vm)]
            if len(ben_pool) < 5:
                ben_pool = pool[pool["label"] == 0]
            ben = take(ben_pool, 5)
            for w, (_, src) in enumerate(mal.iterrows()):
                rows.append(
                    _event(
                        src, vid, w, True, 0, primary_attack, strong_thr, weak_thr,
                        scenario, seed, coordinated=True,
                    )
                )
            for w, (_, src) in enumerate(ben.iterrows(), start=5):
                rows.append(
                    _event(
                        src, vid, w, False, 0, "benign_on_attacked", strong_thr, weak_thr,
                        scenario, seed, coordinated=True,
                    )
                )

        for j in range(n_vehicles - campaign_size):
            vid = f"ben_{j:02d}"
            chosen = take(pool[pool["label"] == 0], windows_per_vehicle)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _event(
                        src, vid, w, False, -1, "benign", strong_thr, weak_thr,
                        scenario, seed, coordinated=False,
                    )
                )

        df = pd.DataFrame(rows)
        cols = [c for c in BEHAVIOURAL_24 if c in df.columns and c in proto_src.columns]
        try:
            proto = compute_campaign_prototype(
                proto_src, attack_type=primary_attack, feature_columns=cols
            )
        except ValueError:
            atk = str(
                proto_src.loc[proto_src["label"] == 1, "attack_type"]
                .value_counts()
                .index[0]
            )
            proto = compute_campaign_prototype(
                proto_src, attack_type=atk, feature_columns=cols
            )
            notes["sampling_adjustments"].append(f"prototype attack_type→{atk}")
        mask = df["scenario_role"] == "coordinated"
        df = apply_coordination_strength(
            df,
            strength=float(coordination_strength),
            campaign_prototype=proto,
            target_mask=mask,
            feature_columns=cols,
            seed=seed,
        )
        df = _derive_gnn9(df)
        notes["n_nodes"] = len(df)
        notes["n_campaign_members"] = int((df["gt_campaign_id"] == 0).sum())
        notes["prototype_blend"] = True
        return df, notes

    if scenario == "unrelated_incidents":
        atk = pool[pool["label"] == 1]
        if len(atk) < campaign_size * 5:
            raise RuntimeError("Insufficient malicious TEST windows for unrelated")
        families = list(atk["attack_type"].value_counts().index)
        for i in range(campaign_size):
            vid = f"ind_{i:02d}"
            fam = families[i % len(families)]
            fam_pool = atk[atk["attack_type"] == fam]
            if len(fam_pool) < 5:
                fam_pool = atk
            chosen = take(fam_pool, 5)
            ben = take(pool[pool["label"] == 0], 5)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _event(
                        src, vid, w, True, i, fam, strong_thr, weak_thr,
                        scenario, seed, coordinated=False,
                    )
                )
            for w, (_, src) in enumerate(ben.iterrows(), start=5):
                rows.append(
                    _event(
                        src, vid, w, False, -1, "benign", strong_thr, weak_thr,
                        scenario, seed, coordinated=False,
                    )
                )
        for j in range(n_vehicles - campaign_size):
            vid = f"ben_{j:02d}"
            chosen = take(pool[pool["label"] == 0], windows_per_vehicle)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _event(
                        src, vid, w, False, -1, "benign", strong_thr, weak_thr,
                        scenario, seed, coordinated=False,
                    )
                )
        df = pd.DataFrame(rows)
        df = _derive_gnn9(df)
        notes["prototype_blend"] = False
        notes["n_nodes"] = len(df)
        return df, notes

    raise ValueError(scenario)
