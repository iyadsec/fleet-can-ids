#!/usr/bin/env python3
"""Publication fleet benign scaler for graph cosine (corrected ablation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

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

DEFAULT_SCALER = (
    Path(__file__).resolve().parents[2]
    / "recovered_publication_pipeline"
    / "new_experiments"
    / "final_end_to_end_publication_run_balanced"
    / "scalers"
    / "fleet_benign_scaler.json"
)


def load_fleet_benign_scaler(path: Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_SCALER
    data = json.loads(path.read_text(encoding="utf-8"))
    names = list(data["fitted_feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(
            "fleet_benign_scaler feature names/order do not match FEATURE_NAMES:\n"
            f"  scaler={names}\n  expected={FEATURE_NAMES}"
        )
    return data


def apply_fleet_scaler_matrix(
    X: np.ndarray,
    scaler: dict,
    *,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> np.ndarray:
    """Z-score columns with publication benign-train means/stds. Never refits."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(f"Expected X shape (n, {len(feature_names)}), got {X.shape}")
    out = np.empty_like(X, dtype=np.float64)
    for j, name in enumerate(feature_names):
        mean = float(scaler["means"][name])
        std = float(scaler["stds"][name])
        out[:, j] = (X[:, j] - mean) / (std + 1e-9)
    return out


def verify_scaler_compatibility(path: Path | None = None) -> dict:
    scaler = load_fleet_benign_scaler(path)
    return {
        "ok": True,
        "scaler_id": scaler.get("scaler_id"),
        "n_features": len(scaler["fitted_feature_names"]),
        "fitted_feature_names": scaler["fitted_feature_names"],
        "matches_feature_names": scaler["fitted_feature_names"] == FEATURE_NAMES,
        "fit_row_count": scaler.get("fit_row_count"),
        "attack_labels_used": scaler.get("attack_labels_used"),
        "training_split": scaler.get("training_split"),
    }
