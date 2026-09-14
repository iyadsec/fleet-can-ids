"""Behavioural coordination-strength dial (no temporal synchronization)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_similarity_features import prepare_fleet_similarity_matrix, SimilarityFeatureView


def _feature_columns(descriptors: pd.DataFrame) -> list[str]:
    return [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in descriptors.columns]


def compute_campaign_prototype(
    descriptors: pd.DataFrame,
    *,
    attack_type: str,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Mean behavioural vector for one attack family (dataset-derived prototype)."""
    cols = feature_columns or _feature_columns(descriptors)
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blend malicious descriptor features toward a shared behavioural prototype.

    strength=0 keeps originals; strength=1 moves fully to prototype (clamped to
  observed per-feature min/max in the source attack subset).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    cols = feature_columns or _feature_columns(descriptors)
    out = descriptors.copy()
    provenance_rows: list[dict] = []

    target_idx = out.index[target_mask]
    if len(target_idx) == 0 or strength <= 0.0:
        return out, pd.DataFrame(
            columns=["event_id", "coordination_strength", "method", "mean_pairwise_similarity_delta"]
        )

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
        provenance_rows.append(
            {
                "event_id": out.loc[idx, "event_id"],
                "coordination_strength": strength,
                "method": "prototype_blend_with_bounded_noise",
                "original_anomaly_score": float(out.loc[idx, "anomaly_score"]),
            }
        )

    provenance = pd.DataFrame(provenance_rows)
    return out, provenance


def measure_mean_pairwise_similarity(
    descriptors: pd.DataFrame,
    mask: pd.Series,
    *,
    feature_view: SimilarityFeatureView = "behavior_only_vehicle_normalized",
    fleet_scaler_provenance: Any | None = None,
) -> float:
    """Mean cosine similarity among masked rows (validation of coordination dial)."""
    sub = descriptors.loc[mask].reset_index(drop=True)
    if len(sub) < 2:
        return 1.0
    scaler = fleet_scaler_provenance
    if scaler is None and feature_view in (
        "behavior_only_vehicle_normalized",
        "behavior_only_locally_normalized",
    ):
        from pathlib import Path

        from src.experiments.local_descriptor_normalisation import load_scaler_provenance
        from src.utils.paths import resolve_project_root

        cache = resolve_project_root() / "new_experiments/metadata_correction/manifests/fleet_benign_scaler.json"
        if cache.exists():
            scaler = load_scaler_provenance(cache)
    X, _, _, _ = prepare_fleet_similarity_matrix(
        sub,
        similarity_feature_view=feature_view,
        feature_dominance_threshold=5.0,
        allowed_high_dominance_features=frozenset(),
        fleet_scaler_provenance=scaler,
    )
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    Xn = X / norms
    sim = Xn @ Xn.T
    triu = sim[np.triu_indices(len(sub), k=1)]
    return float(np.mean(triu)) if triu.size else 0.0
