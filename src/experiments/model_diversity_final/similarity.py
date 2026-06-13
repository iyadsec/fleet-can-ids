"""Cosine-bounded descriptor similarity metrics for final Phase 4."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.experiments.fleet_scaler_loader import resolve_fleet_scaler_from_config
from src.experiments.model_diversity_final.edge_utils import edge_endpoints
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix

COSINE_TOLERANCE = 1e-6


def _l2_rows(X: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(X, dtype=np.float64), norm="l2", axis=1)


def _mean_cosine_pairs(X: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray | None = None) -> float:
    if len(idx_a) < 2 and idx_b is None:
        return float("nan")
    Xn = _l2_rows(X)
    if idx_b is None:
        sub = Xn[idx_a]
        if len(sub) < 2:
            return float("nan")
        sim = cosine_similarity(sub)
        iu = np.triu_indices(len(sub), k=1)
        return float(np.mean(sim[iu])) if len(iu[0]) else float("nan")
    if len(idx_a) == 0 or len(idx_b) == 0:
        return float("nan")
    sim = cosine_similarity(Xn[idx_a], Xn[idx_b])
    return float(np.mean(sim))


def compute_cosine_descriptor_similarity_metrics(
    scenario_df: pd.DataFrame,
    edge_list: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    """Cosine similarity on L2-normalized fleet feature vectors — values in [-1, 1]."""
    if scenario_df.empty:
        return {}
    scaler = resolve_fleet_scaler_from_config(config)
    X, feat_cols = resolve_fleet_similarity_matrix(
        scenario_df,
        similarity_feature_view=config.get("graph", {}).get("similarity_feature_view", "behavior_only_vehicle_normalized"),
        fleet_scaler_provenance=scaler,
    )
    X = np.asarray(X, dtype=np.float64)
    meta = scenario_df.reset_index(drop=True)
    models = meta["vehicle_model"].astype(str).to_numpy()
    mal = meta["ground_truth_malicious"].to_numpy()
    camp = meta["ground_truth_campaign_member"].to_numpy()

    within_model_mal, cross_model_mal, benign_cross = [], [], []
    for model in np.unique(models):
        within_model_mal.append(_mean_cosine_pairs(X, np.where((models == model) & (mal == 1))[0]))
    mal_mask = mal == 1
    umodels = np.unique(models)
    for i, m1 in enumerate(umodels):
        for m2 in umodels[i + 1 :]:
            cross_model_mal.append(
                _mean_cosine_pairs(X, np.where((models == m1) & mal_mask)[0], np.where((models == m2) & mal_mask)[0])
            )
            benign_cross.append(
                _mean_cosine_pairs(X, np.where((models == m1) & (mal == 0))[0], np.where((models == m2) & (mal == 0))[0])
            )

    cross_edges = same_edges = 0
    if not edge_list.empty:
        id_to_model = scenario_df.set_index("event_id")["vehicle_model"].to_dict()
        src, tgt = edge_endpoints(edge_list)
        for s, t in zip(src, tgt):
            m1, m2 = id_to_model.get(s), id_to_model.get(t)
            if m1 and m2:
                if m1 != m2:
                    cross_edges += 1
                else:
                    same_edges += 1
    total_e = cross_edges + same_edges
    return {
        "metric_name": "cosine_similarity_l2_normalized",
        "within_model_attack_similarity": float(np.nanmean(within_model_mal)),
        "cross_model_attack_similarity": float(np.nanmean(cross_model_mal)),
        "campaign_similarity": _mean_cosine_pairs(X, np.where(camp.astype(bool))[0]),
        "benign_cross_model_similarity": float(np.nanmean(benign_cross)),
        "malicious_minus_benign_cross_sim": float(np.nanmean(cross_model_mal) - np.nanmean(benign_cross)),
        "cross_model_edges": cross_edges,
        "same_model_edges": same_edges,
        "cross_model_edge_percentage": 100.0 * cross_edges / total_e if total_e else 0.0,
        "feature_columns": "|".join(feat_cols),
    }


def similarity_diagnostics_row(
    run_id: str,
    scenario_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    scaler = resolve_fleet_scaler_from_config(config)
    X, _ = resolve_fleet_similarity_matrix(
        scenario_df,
        similarity_feature_view=config.get("graph", {}).get("similarity_feature_view", "behavior_only_vehicle_normalized"),
        fleet_scaler_provenance=scaler,
    )
    Xn = _l2_rows(X)
    n = len(Xn)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(np.dot(Xn[i], Xn[j])))
    arr = np.array(pairs) if pairs else np.array([0.0])
    mn, mx = float(arr.min()), float(arr.max())
    ok = bool(mn >= -1.0 - COSINE_TOLERANCE and mx <= 1.0 + COSINE_TOLERANCE)
    return {
        "run_id": run_id,
        "metric_name": "cosine_similarity",
        "minimum": mn,
        "maximum": mx,
        "mean": float(arr.mean()),
        "standard_deviation": float(arr.std()),
        "pair_count": int(len(arr)),
        "within_valid_range": ok,
        "validation_status": "pass" if ok else "fail",
    }
