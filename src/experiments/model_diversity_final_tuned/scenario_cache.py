"""Cache graph/cluster outputs for fast gate grid search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.experiment_pipeline import (
    MethodOutputs,
    _cluster_features,
    _decisions_to_predictions,
    _qualify_clusters_ieee,
    build_scenario_graph,
    resolve_vehicle_id_column,
)
from src.experiments.fleet_scaler_loader import resolve_fleet_scaler_from_config
from src.evaluation.final_gnn_fleet_decision_experiment import run_gnn_fleet_correlation
from src.experiments.model_diversity_final_tuned.tuned_gate import apply_tuned_gate, TunedGateConfig
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


@dataclass
class ScenarioCache:
    run_id: str
    scenario: str
    method: str
    meta: pd.DataFrame
    labels: np.ndarray
    cluster_df: pd.DataFrame
    campaign_scores: np.ndarray
    edge_list: pd.DataFrame
    membership: pd.DataFrame
    expect_campaign: bool


def build_scenario_cache(
    scenario_df: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    method: str,
    *,
    run_id: str,
    scenario: str,
    expect_campaign: bool,
) -> ScenarioCache:
    from src.experiments.experiment_pipeline import _fleet_config_from_experiment, _train_gcn_structure

    cfg = _fleet_config_from_experiment(config, seed)
    gbuild = build_scenario_graph(scenario_df, config, seed)
    meta = gbuild.meta
    scaler = resolve_fleet_scaler_from_config(config)

    if method == "descriptor_clustering":
        X, _ = resolve_fleet_similarity_matrix(
            meta,
            similarity_feature_view=config.get("graph", {}).get("similarity_feature_view", "behavior_only_vehicle_normalized"),
            fleet_scaler_provenance=scaler,
        )
        labels = _cluster_features(X, meta, cfg)
        scores = meta["anomaly_score"].to_numpy(dtype=np.float64)
    elif method == "fcgnn":
        emb, campaign_scores, _ = run_gnn_fleet_correlation(
            gbuild.pyg_data, meta["event_id"].astype(str).tolist(), cfg
        )
        labels = _cluster_features(emb, meta, cfg)
        scores = campaign_scores
    else:
        raise ValueError(method)

    cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, scores)
    return ScenarioCache(
        run_id=run_id,
        scenario=scenario,
        method=method,
        meta=meta,
        labels=labels,
        cluster_df=cluster_df,
        campaign_scores=scores,
        edge_list=gbuild.edge_list,
        membership=membership,
        expect_campaign=expect_campaign,
    )


def evaluate_gate_on_cache(cache: ScenarioCache, gate: TunedGateConfig) -> dict[str, float]:
    decisions, cluster_df = apply_tuned_gate(
        cache.meta, cache.labels, cache.cluster_df, cache.campaign_scores, cache.edge_list, gate,
    )
    decisions = decisions.merge(cache.meta[["event_id", "weak_signal"]], on="event_id", how="left")
    event_pred = _decisions_to_predictions(decisions, cache.membership, cache.method)
    metrics = compute_false_campaign_breakdown(
        event_pred, cache.membership, cluster_df,
        scenario=cache.scenario, expect_campaign=cache.expect_campaign,
    )
    return metrics
