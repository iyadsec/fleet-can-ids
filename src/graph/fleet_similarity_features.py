"""Behaviour-focused feature views for fleet graph similarity (not stored descriptors)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.utils.logging import get_logger

logger = get_logger(__name__)

SimilarityFeatureView = Literal[
    "full_descriptor",
    "behavior_only",
    "behavior_only_vehicle_normalized",
]

SIMILARITY_VIEW_LABELS: dict[str, str] = {
    "full_descriptor": "Full Descriptor",
    "behavior_only": "Behaviour-Only",
    "behavior_only_vehicle_normalized": "Behaviour-Only + Vehicle Normalization",
}

# Identity-heavy columns excluded from behaviour graph view by default.
IDENTITY_HEAVY_COLUMNS: frozenset[str] = frozenset(
    {
        "unique_can_id_count",
        "mean_dlc",
        "std_dlc",
        *{c for c in BEHAVIOURAL_FEATURE_COLUMNS if c.startswith("byte_")},
    }
)

# Behaviour-focused candidates (present in descriptor CSV or derived).
BEHAVIOR_GRAPH_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "frame_count",
    "message_rate",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
)

VEHICLE_NORMALIZE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "frame_count",
    "message_rate",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
)


def build_behavior_view_descriptors(descriptors: pd.DataFrame) -> pd.DataFrame:
    """
    Augment descriptors with derived behaviour features for graph similarity only.

    Does not modify or replace the stored full descriptor table.
    """
    out = descriptors.copy()
    if "frame_count" in out.columns:
        out["message_rate"] = out["frame_count"].astype(np.float64)
    if {"std_inter_arrival_time", "mean_inter_arrival_time"}.issubset(out.columns):
        out["burstiness"] = out["std_inter_arrival_time"].astype(np.float64) / (
            out["mean_inter_arrival_time"].astype(np.float64).abs() + 1e-9
        )
    return out


def compute_feature_dominance(
    descriptors: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Per-feature ratio: between-vehicle variance / mean within-vehicle variance.

    High ratios indicate vehicle-identity encoding.
    """
    if "vehicle_model" not in descriptors.columns:
        raise ValueError("Descriptors require vehicle_model for dominance analysis")

    rows: list[dict[str, object]] = []
    for col in columns:
        if col not in descriptors.columns:
            continue
        series = descriptors[col].astype(np.float64)
        grouped = descriptors.assign(_v=series).groupby("vehicle_model")["_v"]
        between_var = float(grouped.mean().var()) if grouped.ngroups > 1 else 0.0
        within_var = float(grouped.var().mean())
        ratio = between_var / (within_var + 1e-9)
        rows.append(
            {
                "feature": col,
                "between_vehicle_variance": between_var,
                "within_vehicle_variance": within_var,
                "dominance_ratio": ratio,
                "auto_exclude": bool(ratio > 5.0),
            }
        )
    return pd.DataFrame(rows)


def _vehicle_zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = out.groupby("vehicle_model")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9)
        )
    return out


def select_behavior_graph_columns(
    descriptors: pd.DataFrame,
    *,
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> tuple[list[str], pd.DataFrame, list[str]]:
    """
    Choose behaviour columns after removing identity-heavy fields and high-dominance features.

    Returns (selected_columns, dominance_df, removed_columns).
    """
    view_df = build_behavior_view_descriptors(descriptors)
    candidates = [c for c in BEHAVIOR_GRAPH_CANDIDATE_COLUMNS if c in view_df.columns]
    dominance = compute_feature_dominance(view_df, candidates)

    removed_identity = sorted(IDENTITY_HEAVY_COLUMNS & set(BEHAVIOURAL_FEATURE_COLUMNS))
    selected: list[str] = []
    auto_excluded: list[str] = []
    for col in candidates:
        if col in IDENTITY_HEAVY_COLUMNS:
            continue
        row = dominance[dominance["feature"] == col]
        if row.empty:
            selected.append(col)
            continue
        ratio = float(row["dominance_ratio"].iloc[0])
        if ratio > feature_dominance_threshold and col not in allowed_high_dominance_features:
            auto_excluded.append(col)
            continue
        selected.append(col)

    if not selected:
        selected = [c for c in candidates if c not in IDENTITY_HEAVY_COLUMNS]
        logger.warning(
            "All behaviour candidates auto-excluded; using fallback columns: %s", selected
        )

    removed = sorted(set(removed_identity) | set(auto_excluded))
    return selected, dominance, removed


def prepare_fleet_similarity_matrix(
    descriptors: pd.DataFrame,
    *,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> tuple[np.ndarray, list[str], pd.DataFrame, list[str]]:
    """
    Build the feature matrix used only for fleet graph cosine similarity.

    Returns (X, column_names, dominance_df, removed_feature_names).
    """
    removed: list[str] = []
    view_df = build_behavior_view_descriptors(descriptors)

    if similarity_feature_view == "full_descriptor":
        cols = list(BEHAVIOURAL_FEATURE_COLUMNS)
        dominance = compute_feature_dominance(view_df, cols)
        X = view_df[cols].to_numpy(dtype=np.float32)
    elif similarity_feature_view in ("behavior_only", "behavior_only_vehicle_normalized"):
        cols, dominance, removed = select_behavior_graph_columns(
            view_df,
            feature_dominance_threshold=feature_dominance_threshold,
            allowed_high_dominance_features=allowed_high_dominance_features,
        )
        work = view_df.copy()
        if similarity_feature_view == "behavior_only_vehicle_normalized":
            norm_cols = [c for c in VEHICLE_NORMALIZE_COLUMNS if c in cols]
            work = _vehicle_zscore(work, norm_cols)
        X = work[cols].to_numpy(dtype=np.float32)
    else:
        raise ValueError(f"Unknown similarity_feature_view: {similarity_feature_view}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    dominance = dominance.copy()
    dominance["similarity_feature_view"] = similarity_feature_view
    logger.info(
        "Fleet similarity view=%s: %d features %s",
        similarity_feature_view,
        len(cols),
        cols,
    )
    return X, cols, dominance, removed


def parse_fleet_graph_similarity_settings(config: dict) -> dict[str, object]:
    """Read fleet_graph similarity settings from YAML config."""
    fg = config.get("fleet_graph", {})
    allowed = fg.get("allowed_high_dominance_features", []) or []
    return {
        "similarity_feature_view": fg.get(
            "similarity_feature_view", "behavior_only_vehicle_normalized"
        ),
        "feature_dominance_threshold": float(fg.get("feature_dominance_threshold", 5.0)),
        "allowed_high_dominance_features": frozenset(str(x) for x in allowed),
        "top_k_neighbors": int(fg.get("top_k_neighbors", 15)),
        "top_k_same_vehicle": int(fg.get("top_k_same_vehicle", 10)),
        "top_k_cross_vehicle": int(fg.get("top_k_cross_vehicle", 5)),
        "similarity_threshold": float(fg.get("similarity_threshold", 0.95)),
    }
