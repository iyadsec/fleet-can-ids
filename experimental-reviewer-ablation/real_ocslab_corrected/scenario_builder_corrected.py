#!/usr/bin/env python3
"""Publication-faithful campaign scenario builder for corrected ablation.

Implements recoverable semantics from:
  coordination_strength.compute_campaign_prototype
  coordination_strength.apply_coordination_strength (strength=1.0)
  campaign_analysis_corrected platform compositions

Does NOT use PR #22 score-band-only grouping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_scaler import FEATURE_NAMES

# Behavioural columns used for prototype blend (24-D manuscript features).
BEHAVIOURAL_24 = [
    "frame_count",
    "unique_can_id_count",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "mean_dlc",
    "std_dlc",
    *[f"byte_mean_{i}" for i in range(8)],
    *[f"byte_std_{i}" for i in range(8)],
]

STRONG_COMPOSITION = {"Hyundai": 2, "Kia": 2, "Chevrolet": 1}
WEAK_COMPOSITION = {"Hyundai": 3, "Kia": 2, "Chevrolet": 0}


def compute_campaign_prototype(
    descriptors: pd.DataFrame,
    *,
    attack_type: str,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    cols = feature_columns or [c for c in BEHAVIOURAL_24 if c in descriptors.columns]
    sub = descriptors[descriptors["attack_type"] == attack_type]
    if sub.empty:
        raise ValueError(f"No rows for attack_type={attack_type}")
    return sub[cols].astype(np.float64).mean(axis=0).to_numpy()


def apply_coordination_strength(
    descriptors: pd.DataFrame,
    *,
    strength: float,
    campaign_prototype: np.ndarray,
    target_mask: pd.Series,
    feature_columns: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Blend toward shared prototype. strength=1.0 → full prototype (clipped)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    cols = feature_columns or [c for c in BEHAVIOURAL_24 if c in descriptors.columns]
    out = descriptors.copy()
    target_idx = out.index[target_mask]
    if len(target_idx) == 0 or strength <= 0.0:
        return out
    sub = out.loc[target_idx, cols].astype(np.float64)
    feat_min = sub.min(axis=0).to_numpy()
    feat_max = sub.max(axis=0).to_numpy()
    proto = np.clip(campaign_prototype, feat_min, feat_max)
    rng = np.random.default_rng(seed)
    noise_scale = 0.02 * (1.0 - strength)
    for idx in target_idx:
        original = out.loc[idx, cols].astype(np.float64).to_numpy()
        blended = (1.0 - strength) * original + strength * proto
        if noise_scale > 0:
            blended += rng.normal(0.0, noise_scale, size=len(cols))
        blended = np.clip(blended, feat_min, feat_max)
        out.loc[idx, cols] = blended
    return out


def _derive_gnn9(df: pd.DataFrame) -> pd.DataFrame:
    """Build 9-D g_i columns from 24-D + anomaly_score (publication behaviour view)."""
    out = df.copy()
    if "message_rate" not in out.columns:
        out["message_rate"] = out["frame_count"].astype(np.float64)
    if "burstiness" not in out.columns:
        out["burstiness"] = out["std_inter_arrival_time"].astype(np.float64) / (
            out["mean_inter_arrival_time"].astype(np.float64).abs() + 1e-9
        )
    if "payload_entropy" not in out.columns:
        # Prefer precomputed; else approximate from byte_std means (diagnostic fallback).
        byte_stds = [c for c in out.columns if c.startswith("byte_std_")]
        if byte_stds:
            out["payload_entropy"] = out[byte_stds].astype(np.float64).mean(axis=1)
        else:
            out["payload_entropy"] = 0.0
    return out


def platform_composition(attack_strength: str, campaign_size: int = 5) -> dict[str, int]:
    if campaign_size != 5:
        raise ValueError("Corrected ablation fixes campaign_size=5")
    if attack_strength == "strong":
        return dict(STRONG_COMPOSITION)
    if attack_strength == "weak":
        return dict(WEAK_COMPOSITION)
    raise ValueError(attack_strength)


def freeze_scenario(df: pd.DataFrame, cache_dir: Path, notes: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    scenario = str(df["scenario"].iloc[0])
    seed = int(df["seed"].iloc[0])
    path = cache_dir / f"{scenario}_seed{seed}.csv"
    df.to_csv(path, index=False)
    (cache_dir / f"{scenario}_seed{seed}_notes.json").write_text(
        json.dumps(notes, indent=2), encoding="utf-8"
    )
    return path


def load_frozen_scenario(cache_dir: Path, scenario: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(cache_dir / f"{scenario}_seed{seed}.csv")


def pack_frozen(df: pd.DataFrame) -> dict[str, Any]:
    df = _derive_gnn9(df)
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    meta = df.copy()
    return {
        "X": X,
        "vehicles": df["vehicle_id"].astype(str).to_numpy(),
        "meta": meta,
        "is_attack": df["is_attack"].astype(int).to_numpy(),
        "gt_campaign_id": df["gt_campaign_id"].astype(int).to_numpy(),
        "descriptor_ids": df["event_id"].astype(str).tolist(),
        "anomaly_score": df["anomaly_score"].astype(float).to_numpy(),
    }
