"""Graph methods with two-level tuned campaign gate (C2/C3)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
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
from src.experiments.model_diversity_final_tuned.tuned_gate import TunedGateConfig, apply_tuned_gate
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


def run_tuned_gated_graph_method(
    descriptors: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    method: str,
    gate: TunedGateConfig,
) -> MethodOutputs:
    from src.experiments.experiment_pipeline import _fleet_config_from_experiment, _train_gcn_structure

    cfg = _fleet_config_from_experiment(config, seed)
    gbuild = build_scenario_graph(descriptors, config, seed)
    meta = gbuild.meta
    runtime: dict[str, float] = {"graph_construction_sec": gbuild.graph_build_sec}
    scaler = resolve_fleet_scaler_from_config(config)

    if method == "descriptor_clustering":
        X, _ = resolve_fleet_similarity_matrix(
            meta,
            similarity_feature_view=config.get("graph", {}).get("similarity_feature_view", "behavior_only_vehicle_normalized"),
            fleet_scaler_provenance=scaler,
        )
        labels = _cluster_features(X, meta, cfg)
        scores = meta["anomaly_score"].to_numpy(dtype=np.float64)
        cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, scores)
        embeddings = None
    elif method == "fcgnn":
        emb, campaign_scores, _ = run_gnn_fleet_correlation(gbuild.pyg_data, meta["event_id"].astype(str).tolist(), cfg)
        scores = campaign_scores
        labels = _cluster_features(emb, meta, cfg)
        cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, scores)
        embeddings = emb
    else:
        raise ValueError(f"Unsupported method: {method}")

    decisions, cluster_df = apply_tuned_gate(meta, labels, cluster_df, scores, gbuild.edge_list, gate)
    decisions = decisions.merge(meta[["event_id", "weak_signal"]], on="event_id", how="left")
    event_pred = _decisions_to_predictions(decisions, membership, method)
    veh_col = resolve_vehicle_id_column(event_pred)
    vehicle_pred = (
        event_pred.groupby(veh_col, as_index=False)
        .agg(
            predicted_attacked=("final_decision", lambda s: int((s == DECISION_COORDINATED).any())),
            ground_truth_attacked=("ground_truth_campaign_member", "max")
            if "ground_truth_campaign_member" in event_pred.columns
            else ("ground_truth_malicious", "max"),
            n_events=("event_id", "count"),
        )
        .rename(columns={veh_col: "vehicle_id"})
    )
    if "ground_truth_campaign_member" in membership.columns:
        gt_map = membership.groupby(veh_col)["ground_truth_campaign_member"].max()
        vehicle_pred["ground_truth_attacked"] = vehicle_pred["vehicle_id"].map(gt_map).fillna(0).astype(int)

    accepted = cluster_df[cluster_df.get("campaign_accepted", False)] if not cluster_df.empty else cluster_df
    campaign_pred = pd.DataFrame(
        [{"method": method, "n_qualifying_campaign_clusters": len(cluster_df[cluster_df["is_qualifying_campaign_cluster"]]) if not cluster_df.empty else 0,
          "n_accepted_campaign_clusters": len(accepted)}]
    )
    return MethodOutputs(
        event_predictions=event_pred,
        vehicle_predictions=vehicle_pred,
        campaign_predictions=campaign_pred,
        cluster_df=cluster_df,
        embeddings=embeddings,
        graph_stats=gbuild.stats,
        edge_list=gbuild.edge_list,
        runtime=runtime,
    )
