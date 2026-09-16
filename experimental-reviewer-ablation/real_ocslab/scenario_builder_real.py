#!/usr/bin/env python3
"""Real-OCSLab controlled fleet scenarios for the Reviewer-A ablation.

Same scenario shape as the synthetic harness (20×10 nodes, campaign size 5,
three scenarios), but samples ONLY held-out TEST windows from the scored
OCSLab pool. No synthetic descriptors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Must match parent experimental-reviewer-ablation/scenario_builder.FEATURE_NAMES
FEATURE_NAMES = [
    "anomaly_score",
    "frame_count",
    "message_rate",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "payload_entropy",
]


def load_test_pool(pool_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pool_csv)
    if "window_uid" not in df.columns and "window_uid" in df.columns:
        df = df.rename(columns={"window_uid": "window_uid"})
    if "window_uid" not in df.columns:
        raise ValueError("TEST pool requires window_uid")
    required = {
        "window_uid",
        "label",
        "anomaly_score",
        "physical_vehicle",
        "source_trace",
        "attack_type",
        *FEATURE_NAMES,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TEST pool missing columns: {sorted(missing)}")
    return df


def pool_inventory(
    pool: pd.DataFrame, *, strong_thr: float = 0.80, weak_thr: float = 0.55
) -> dict[str, Any]:
    strong = pool[(pool["label"] == 1) & (pool["anomaly_score"] >= strong_thr)]
    weak = pool[
        (pool["label"] == 1)
        & (pool["anomaly_score"] >= weak_thr)
        & (pool["anomaly_score"] < strong_thr)
    ]
    sus_benign = pool[(pool["label"] == 0) & (pool["anomaly_score"] >= weak_thr)]
    return {
        "n_test": int(len(pool)),
        "physical_vehicles": sorted(pool["physical_vehicle"].astype(str).unique()),
        "source_traces": sorted(pool["source_trace"].astype(str).unique()),
        "n_strong_malicious": int(len(strong)),
        "n_weak_malicious": int(len(weak)),
        "n_suspicious_benign": int(len(sus_benign)),
        "strong_by_attack": {str(k): int(v) for k, v in strong["attack_type"].value_counts().items()},
        "weak_by_attack": {str(k): int(v) for k, v in weak["attack_type"].value_counts().items()},
        "strong_by_vehicle": {
            str(k): int(v) for k, v in strong["physical_vehicle"].value_counts().items()
        },
        "weak_by_vehicle": {
            str(k): int(v) for k, v in weak["physical_vehicle"].value_counts().items()
        },
    }


def _sample_rows(
    pool: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    *,
    used: set[str],
) -> pd.DataFrame:
    cand = pool[~pool["window_uid"].astype(str).isin(used)]
    replace = False
    if len(cand) < n:
        cand = pool
        replace = True
    if len(cand) == 0:
        raise RuntimeError("Empty candidate pool for scenario sampling")
    idx = rng.choice(len(cand), size=n, replace=replace or len(cand) < n)
    chosen = cand.iloc[idx].copy()
    for uid in chosen["window_uid"].astype(str):
        used.add(uid)
    return chosen


def _row_to_event(
    src: pd.Series,
    *,
    event_id: str,
    vehicle_id: str,
    window_id: int,
    is_attack: bool,
    campaign_id: int,
    attack_family: str,
    strong_thr: float,
    weak_thr: float,
    scenario: str,
    seed: int,
) -> dict[str, Any]:
    score = float(src["anomaly_score"])
    local_alert = int(score >= strong_thr)
    weak_signal = int((not local_alert) and score >= weak_thr)
    if not local_alert and not weak_signal:
        weak_signal = 1
    evidence = "strong_local_anomaly" if local_alert else "weak_suspicious_signal"
    row: dict[str, Any] = {
        "event_id": event_id,
        "window_id": int(window_id),
        "vehicle_id": vehicle_id,
        "vehicle_model": vehicle_id,
        "physical_vehicle": str(src["physical_vehicle"]),
        "source_file": str(src.get("source_file", src["source_trace"])),
        "source_trace": str(src["source_trace"]),
        "source_window_uid": str(src["window_uid"]),
        "attack_type": str(src["attack_type"]),
        "anomaly_score": score,
        "evidence_level": evidence,
        "local_alert": local_alert,
        "weak_signal": weak_signal,
        "is_attack": int(is_attack),
        "gt_campaign_id": int(campaign_id) if is_attack and campaign_id >= 0 else -1,
        "attack_family": attack_family if is_attack else "benign_fp",
        "scenario": scenario,
        "seed": int(seed),
        "synthetic": 0,
    }
    for name in FEATURE_NAMES:
        row[name] = float(src[name])
    row["anomaly_score"] = score
    return row


def _pick_attack_family(pool: pd.DataFrame, rng: np.random.Generator) -> str:
    counts = pool["attack_type"].value_counts()
    families = [f for f, c in counts.items() if c >= 10]
    if not families:
        families = counts.index.tolist()
    if not families:
        raise RuntimeError("No attack families available in candidate pool")
    return str(rng.choice(families))


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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    offset = {
        "strong_campaign": 1,
        "weak_campaign": 2,
        "unrelated_incidents": 3,
    }[scenario]
    rng = np.random.default_rng(seed + offset * 10_000)
    vehicles = [f"veh_{i:02d}" for i in range(n_vehicles)]
    used: set[str] = set()
    notes: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "virtual_fleet": True,
        "n_physical_vehicles_in_pool": int(pool["physical_vehicle"].nunique()),
        "sampling_adjustments": [],
    }

    strong_mal = pool[(pool["label"] == 1) & (pool["anomaly_score"] >= strong_thr)]
    weak_mal = pool[
        (pool["label"] == 1)
        & (pool["anomaly_score"] >= weak_thr)
        & (pool["anomaly_score"] < strong_thr)
    ]
    sus_benign = pool[(pool["label"] == 0) & (pool["anomaly_score"] >= weak_thr)]
    if sus_benign.empty:
        benign = pool[pool["label"] == 0].copy()
        if benign.empty:
            raise RuntimeError("No benign TEST windows available")
        sus_benign = benign.nlargest(min(len(benign), 500), "anomaly_score")
        notes["sampling_adjustments"].append(
            "suspicious_benign_empty: used top-scoring benign TEST windows"
        )

    cs = campaign_size
    wpv = windows_per_vehicle

    def add_background(rows: list[dict[str, Any]]) -> None:
        for vid in vehicles[cs:]:
            chosen = _sample_rows(sus_benign, wpv, rng, used=used)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _row_to_event(
                        src,
                        event_id=f"{vid}-w{w:02d}",
                        vehicle_id=vid,
                        window_id=w,
                        is_attack=False,
                        campaign_id=-1,
                        attack_family="benign_fp",
                        strong_thr=strong_thr,
                        weak_thr=weak_thr,
                        scenario=scenario,
                        seed=seed,
                    )
                )

    if scenario == "strong_campaign":
        if strong_mal.empty:
            raise RuntimeError("No strong malicious TEST windows")
        if len(strong_mal) < cs * wpv:
            notes["sampling_adjustments"].append(
                f"strong_pool_size={len(strong_mal)} < {cs * wpv}; sampling with replacement"
            )
        family = _pick_attack_family(strong_mal, rng)
        family_pool = strong_mal[strong_mal["attack_type"] == family]
        if len(family_pool) < wpv:
            family_pool = strong_mal
            notes["sampling_adjustments"].append(
                f"strong family {family} too small; using all strong attacks"
            )
        rows: list[dict[str, Any]] = []
        for vid in vehicles[:cs]:
            chosen = _sample_rows(family_pool, wpv, rng, used=used)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _row_to_event(
                        src,
                        event_id=f"{vid}-w{w:02d}",
                        vehicle_id=vid,
                        window_id=w,
                        is_attack=True,
                        campaign_id=0,
                        attack_family=str(family),
                        strong_thr=strong_thr,
                        weak_thr=weak_thr,
                        scenario=scenario,
                        seed=seed,
                    )
                )
        add_background(rows)
        df = pd.DataFrame(rows)
        notes["campaign_attack_family"] = family

    elif scenario == "weak_campaign":
        if weak_mal.empty:
            raise RuntimeError(
                "No weak malicious TEST windows "
                f"(need {weak_thr} <= anomaly_score < {strong_thr}). "
                "Refusing to fabricate weak observations."
            )
        if len(weak_mal) < cs * wpv:
            notes["sampling_adjustments"].append(
                f"weak_pool_size={len(weak_mal)} < {cs * wpv}; sampling with replacement "
                "(scores unchanged; no fabrication)"
            )
        family = _pick_attack_family(weak_mal, rng)
        family_pool = weak_mal[weak_mal["attack_type"] == family]
        if len(family_pool) < wpv:
            family_pool = weak_mal
            notes["sampling_adjustments"].append(
                f"weak family {family} too small; using all weak-band attacks"
            )
        rows = []
        for vid in vehicles[:cs]:
            chosen = _sample_rows(family_pool, wpv, rng, used=used)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _row_to_event(
                        src,
                        event_id=f"{vid}-w{w:02d}",
                        vehicle_id=vid,
                        window_id=w,
                        is_attack=True,
                        campaign_id=0,
                        attack_family=str(family),
                        strong_thr=strong_thr,
                        weak_thr=weak_thr,
                        scenario=scenario,
                        seed=seed,
                    )
                )
        add_background(rows)
        df = pd.DataFrame(rows)
        notes["campaign_attack_family"] = family

    elif scenario == "unrelated_incidents":
        attack_pool = (
            strong_mal
            if len(strong_mal) >= cs
            else pd.concat([strong_mal, weak_mal], ignore_index=True)
        )
        if attack_pool.empty:
            raise RuntimeError("No malicious TEST windows for unrelated incidents")
        families = attack_pool["attack_type"].value_counts().index.tolist()
        if len(families) < cs:
            notes["sampling_adjustments"].append(
                f"only {len(families)} attack families; cycling for {cs} vehicles"
            )
        rows = []
        for i, vid in enumerate(vehicles[:cs]):
            fam = str(families[i % len(families)])
            fam_pool = attack_pool[attack_pool["attack_type"] == fam]
            if fam_pool.empty:
                fam_pool = attack_pool
            chosen = _sample_rows(fam_pool, wpv, rng, used=used)
            for w, (_, src) in enumerate(chosen.iterrows()):
                rows.append(
                    _row_to_event(
                        src,
                        event_id=f"{vid}-w{w:02d}",
                        vehicle_id=vid,
                        window_id=w,
                        is_attack=True,
                        campaign_id=i,
                        attack_family=f"independent_{i}_{fam}",
                        strong_thr=strong_thr,
                        weak_thr=weak_thr,
                        scenario=scenario,
                        seed=seed,
                    )
                )
        add_background(rows)
        df = pd.DataFrame(rows)
        notes["independent_families"] = [
            str(families[i % len(families)]) for i in range(cs)
        ]
    else:
        raise ValueError(scenario)

    df = df.sort_values(["vehicle_id", "window_id"]).reset_index(drop=True)
    assert len(df) == n_vehicles * windows_per_vehicle
    assert int(df["synthetic"].sum()) == 0
    notes["n_unique_source_window_uids"] = int(df["source_window_uid"].nunique())
    notes["physical_vehicle_counts"] = {
        str(k): int(v) for k, v in df["physical_vehicle"].value_counts().items()
    }
    return df, notes


def freeze_scenario(
    df: pd.DataFrame, cache_dir: Path, notes: dict[str, Any] | None = None
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenario = str(df["scenario"].iloc[0])
    seed = int(df["seed"].iloc[0])
    path = cache_dir / f"{scenario}_seed{seed}.csv"
    df.to_csv(path, index=False)
    meta = {
        "scenario": scenario,
        "seed": seed,
        "n_nodes": int(len(df)),
        "descriptor_ids": df["event_id"].tolist(),
        "vehicle_ids": sorted(df["vehicle_id"].unique().tolist()),
        "gt_campaign_ids": sorted(
            {int(c) for c in df["gt_campaign_id"].tolist() if int(c) >= 0}
        ),
        "feature_names": FEATURE_NAMES,
        "synthetic": False,
        "notes": notes or {},
    }
    (cache_dir / f"{scenario}_seed{seed}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path


def load_frozen_scenario(cache_dir: Path, scenario: str, seed: int) -> pd.DataFrame:
    path = cache_dir / f"{scenario}_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pack_frozen(df: pd.DataFrame) -> dict[str, Any]:
    """Match parent pack_frozen keys expected by methods.py / run_ablation.py."""
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    vehicles = df["vehicle_id"].astype(str).tolist()
    gt = df["gt_campaign_id"].astype(int).to_numpy()
    return {
        "df": df,
        "X": X,
        "vehicles": vehicles,
        "meta": df.copy(),
        "descriptor_ids": df["event_id"].astype(str).tolist(),
        "gt_campaign_id": gt,
        "is_attack": df["is_attack"].to_numpy(dtype=bool),
        "scenario": str(df["scenario"].iloc[0]),
        "seed": int(df["seed"].iloc[0]),
    }
