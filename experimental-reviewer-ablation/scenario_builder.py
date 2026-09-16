"""Synthetic controlled fleet scenarios for the Reviewer-A ablation.

Each scenario yields exactly 200 suspicious behavioural descriptors
(20 vehicles × 10 windows), matching FLEET-GUARD practice of feeding only
suspicious / weak-signal windows into fleet correlation.

Absolute metrics are NOT claimed identical to frozen historical publication tables.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _proto(kind: str, family: int = 0) -> np.ndarray:
    if kind == "attack_shared":
        return np.array(
            [0.90, 100.0, 2.5, 0.75, 0.004, 0.006, 2.0, 0.55, 4.5], dtype=np.float64
        )
    if kind == "attack_indep":
        bases = [
            np.array([0.88, 100.0, 1.8, 0.55, 0.006, 0.008, 4.2, 0.35, 5.0]),
            np.array([0.87, 100.0, 3.2, 0.90, 0.003, 0.010, 1.5, 0.70, 3.8]),
            np.array([0.89, 100.0, 2.1, 0.60, 0.005, 0.007, 2.8, 0.45, 4.8]),
            np.array([0.86, 100.0, 2.8, 0.85, 0.003, 0.009, 1.8, 0.65, 5.2]),
            np.array([0.91, 100.0, 2.2, 0.50, 0.007, 0.005, 3.5, 0.40, 4.0]),
        ]
        return bases[family % len(bases)].astype(np.float64)
    if kind == "benign_fp":
        base = np.array(
            [0.62, 100.0, 1.1, 0.20, 0.010, 0.003, 3.4, 0.28, 3.3], dtype=np.float64
        )
        offset = np.zeros(9, dtype=np.float64)
        offset[2] = 0.15 * (family % 7)
        offset[3] = 0.05 * (family % 5)
        offset[4] = 0.002 * ((family * 3) % 6)
        offset[6] = 0.25 * (family % 9)
        offset[7] = 0.04 * (family % 6)
        offset[8] = 0.20 * (family % 8)
        out = base + offset
        out[0] = 0.55 + 0.03 * (family % 4)
        out[1] = 100.0
        return out
    raise ValueError(kind)


def _rows_for_vehicles(
    *,
    vehicle_ids: list[str],
    proto: np.ndarray,
    noise: float,
    anomaly_shift: float,
    rng: np.random.Generator,
    windows: int,
    is_attack: bool,
    campaign_id: int,
    attack_family: str,
    strong_thr: float = 0.80,
    weak_thr: float = 0.55,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for vid in vehicle_ids:
        for w in range(windows):
            x = proto + rng.normal(0.0, noise, size=len(FEATURE_NAMES))
            x[0] = float(np.clip(x[0] + anomaly_shift, 0.0, 1.0))
            x[1] = 100.0
            local_alert = int(x[0] >= strong_thr)
            weak_signal = int((not local_alert) and x[0] >= weak_thr)
            if not local_alert and not weak_signal:
                weak_signal = 1
                x[0] = max(float(x[0]), weak_thr)
            evidence = (
                "strong_local_anomaly" if local_alert else "weak_suspicious_signal"
            )
            row: dict[str, Any] = {
                "event_id": f"{vid}-w{w:02d}",
                "window_id": int(w),
                "vehicle_id": vid,
                "vehicle_model": vid,
                "source_file": f"synthetic/{vid}.csv",
                "attack_type": attack_family if is_attack else "benign_fp",
                "anomaly_score": float(x[0]),
                "evidence_level": evidence,
                "local_alert": local_alert,
                "weak_signal": weak_signal,
                "is_attack": int(is_attack),
                "gt_campaign_id": (
                    int(campaign_id) if is_attack and campaign_id >= 0 else -1
                ),
                "attack_family": attack_family if is_attack else "benign_fp",
            }
            for i, name in enumerate(FEATURE_NAMES):
                row[name] = float(x[i])
            row["anomaly_score"] = float(x[0])
            rows.append(row)
    return rows


def build_scenario(
    scenario: str,
    seed: int,
    *,
    n_vehicles: int = 20,
    windows_per_vehicle: int = 10,
    campaign_size: int = 5,
) -> pd.DataFrame:
    offset = {
        "strong_campaign": 1,
        "weak_campaign": 2,
        "unrelated_incidents": 3,
    }[scenario]
    rng = np.random.default_rng(seed + offset * 10_000)
    vehicles = [f"veh_{i:02d}" for i in range(n_vehicles)]

    if scenario in ("strong_campaign", "weak_campaign"):
        attacked = vehicles[:campaign_size]
        rest = vehicles[campaign_size:]
        if scenario == "strong_campaign":
            noise, shift = 0.015, 0.0
        else:
            noise, shift = 0.06, -0.22
        rows: list[dict[str, Any]] = []
        rows += _rows_for_vehicles(
            vehicle_ids=attacked,
            proto=_proto("attack_shared"),
            noise=noise,
            anomaly_shift=shift,
            rng=rng,
            windows=windows_per_vehicle,
            is_attack=True,
            campaign_id=0,
            attack_family="coordinated_dos",
        )
        for j, vid in enumerate(rest):
            rows += _rows_for_vehicles(
                vehicle_ids=[vid],
                proto=_proto("benign_fp", family=j + 1),
                noise=0.04,
                anomaly_shift=0.0,
                rng=rng,
                windows=windows_per_vehicle,
                is_attack=False,
                campaign_id=-1,
                attack_family="benign_fp",
            )
        df = pd.DataFrame(rows)
    elif scenario == "unrelated_incidents":
        rows = []
        for i, vid in enumerate(vehicles[:campaign_size]):
            rows += _rows_for_vehicles(
                vehicle_ids=[vid],
                proto=_proto("attack_indep", family=i),
                noise=0.025,
                anomaly_shift=0.0,
                rng=rng,
                windows=windows_per_vehicle,
                is_attack=True,
                campaign_id=i,
                attack_family=f"independent_{i}",
            )
        for j, vid in enumerate(vehicles[campaign_size:]):
            rows += _rows_for_vehicles(
                vehicle_ids=[vid],
                proto=_proto("benign_fp", family=j + 10),
                noise=0.04,
                anomaly_shift=0.0,
                rng=rng,
                windows=windows_per_vehicle,
                is_attack=False,
                campaign_id=-1,
                attack_family="benign_fp",
            )
        df = pd.DataFrame(rows)
    else:
        raise ValueError(scenario)

    df["scenario"] = scenario
    df["seed"] = int(seed)
    df = df.sort_values(["vehicle_id", "window_id"]).reset_index(drop=True)
    assert len(df) == n_vehicles * windows_per_vehicle
    assert ((df["local_alert"] == 1) | (df["weak_signal"] == 1)).all()
    return df


def freeze_scenario(df: pd.DataFrame, cache_dir: Path) -> Path:
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
