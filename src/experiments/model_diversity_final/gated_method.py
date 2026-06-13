"""Graph methods with validation-tuned campaign membership gate (C2/C3 shared semantics)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED, DECISION_ISOLATED
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
from src.experiments.model_diversity_final.campaign_gate import CampaignGateConfig, gate_qualifying_clusters
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


def _assign_gated_decisions(
    meta: pd.DataFrame,
    labels: np.ndarray,
    cluster_df: pd.DataFrame,
    campaign_scores: np.ndarray,
    edge_list: pd.DataFrame,
    gate: CampaignGateConfig,
) -> pd.DataFrame:
    qualifying = {
        int(r["cluster_id"])
        for _, r in cluster_df.iterrows()
        if bool(r["is_qualifying_campaign_cluster"])
    }
    gated = gate_qualifying_clusters(cluster_df, labels, meta, edge_list, gate)
    cluster_info = cluster_df.set_index("cluster_id") if not cluster_df.empty else pd.DataFrame()
    rows = []
    meta_r = meta.reset_index(drop=True)
    for i, row in meta_r.iterrows():
        cid = int(labels[i])
        in_qual = cid in qualifying
        in_gate = cid in gated
        local_strong = int(row.get("local_alert", 0)) == 1
        score = float(campaign_scores[i])
        member = in_qual and in_gate and (local_strong or score >= gate.min_membership_confidence)
        info = cluster_info.loc[cid] if cid in cluster_info.index else None
        rows.append(
            {
                "window_id": int(row["window_id"]),
                "vehicle_id": row["vehicle_model"],
                "vehicle_model": row["vehicle_model"],
                "event_id": row["event_id"],
                "attack_type": row["attack_type"],
                "anomaly_score": float(row["anomaly_score"]),
                "local_alert": int(row.get("local_alert", 0)),
                "gnn_campaign_score": round(score, 4),
                "cluster_id": cid,
                "vehicles_in_cluster": int(info["vehicles_in_cluster"]) if info is not None else 0,
                "behavioral_cohesion": round(float(info["behavioral_cohesion"]), 4) if info is not None else 0.0,
                "eval_dominant_attack_type": str(info["eval_dominant_attack_type"]) if info is not None else "",
                "final_decision": DECISION_COORDINATED if member else DECISION_ISOLATED,
                "campaign_gate_passed": int(in_gate),
            }
        )
    return pd.DataFrame(rows)


def run_gated_graph_method(
    descriptors: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    method: str,
    gate: CampaignGateConfig,
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
    else:
        if method == "fcgnn":
            emb, campaign_scores, _ = run_gnn_fleet_correlation(gbuild.pyg_data, meta["event_id"].astype(str).tolist(), cfg)
            scores = campaign_scores
        else:
            emb, campaign_scores, _ = _train_gcn_structure(gbuild.pyg_data, cfg)
            scores = campaign_scores
        labels = _cluster_features(emb, meta, cfg)
        cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, scores)
        embeddings = emb if method == "fcgnn" else None

    decisions = _assign_gated_decisions(meta, labels, cluster_df, scores, gbuild.edge_list, gate)
    decisions = decisions.merge(meta[["event_id", "weak_signal"]], on="event_id", how="left")
    event_pred = _decisions_to_predictions(decisions, membership, method)
    veh_col = resolve_vehicle_id_column(event_pred)
    vehicle_pred = (
        event_pred.groupby(veh_col, as_index=False)
        .agg(
            predicted_attacked=("final_decision", lambda s: int((s == DECISION_COORDINATED).any())),
            ground_truth_attacked=("ground_truth_campaign_member", "max") if "ground_truth_campaign_member" in event_pred.columns else ("ground_truth_malicious", "max"),
            n_events=("event_id", "count"),
        )
        .rename(columns={veh_col: "vehicle_id"})
    )
    if "ground_truth_campaign_member" in membership.columns:
        gt_map = membership.groupby(veh_col)["ground_truth_campaign_member"].max()
        vehicle_pred["ground_truth_attacked"] = vehicle_pred["vehicle_id"].map(gt_map).fillna(0).astype(int)

    qualifying = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else cluster_df
    campaign_pred = pd.DataFrame(
        [{"method": method, "n_qualifying_campaign_clusters": len(qualifying), "n_gated_clusters": int(decisions["campaign_gate_passed"].sum())}]
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
