"""Shared graph build, clustering, and campaign decision for M2–M4."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from src.evaluation.campaign_clustering import extend_dbscan_labels, run_dbscan, subsample_indices
from src.evaluation.final_gnn_fleet_decision_experiment import (
    DECISION_COORDINATED,
    DECISION_ISOLATED,
    FinalGnnFleetConfig,
    assign_final_decisions,
    build_final_gnn_fleet_graph,
    cluster_gnn_embeddings,
    compute_cluster_behavioral_cohesion,
    prepare_gnn_fleet_node_matrix,
    run_gnn_fleet_correlation,
)
from src.graph.fleet_graph_builder import (
    build_networkx_graph,
    build_pyg_data,
    graph_to_tables,
    resolve_fleet_similarity_matrix,
)
from src.models.gnn_models import GraphSAGEFleetCorrelator, train_graphsage_fleet_correlation

MethodId = Literal["descriptor_clustering", "standard_gnn", "fcgnn"]


from src.experiments.fleet_scaler_loader import resolve_fleet_scaler_from_config
from src.experiments.vehicle_identity import resolve_vehicle_id_column


@dataclass
class GraphBuildResult:
    graph: nx.Graph
    pyg_data: Any
    stats: pd.DataFrame
    behavior_features: np.ndarray
    meta: pd.DataFrame
    edge_list: pd.DataFrame
    graph_build_sec: float


@dataclass
class MethodOutputs:
    event_predictions: pd.DataFrame
    vehicle_predictions: pd.DataFrame
    campaign_predictions: pd.DataFrame
    cluster_df: pd.DataFrame
    embeddings: np.ndarray | None
    graph_stats: pd.DataFrame
    edge_list: pd.DataFrame
    runtime: dict[str, float]


def _fleet_config_from_experiment(config: dict[str, Any], seed: int) -> FinalGnnFleetConfig:
    camp = config.get("campaign", {})
    gnn = config.get("gnn", {})
    graph = config.get("graph", {})
    return FinalGnnFleetConfig(
        top_k_same_vehicle=int(graph.get("top_k_same_vehicle", 10)),
        top_k_cross_vehicle=int(graph.get("top_k_cross_vehicle", 5)),
        similarity_threshold=float(graph.get("default_similarity_threshold", 0.95)),
        gnn_hidden_channels=int(gnn.get("hidden_channels", 64)),
        gnn_embedding_dim=int(gnn.get("embedding_dim", 32)),
        gnn_epochs=int(gnn.get("epochs", 30)),
        gnn_learning_rate=float(gnn.get("learning_rate", 0.01)),
        gnn_weight_decay=float(gnn.get("weight_decay", 5e-4)),
        gnn_train_ratio=float(gnn.get("train_ratio", 0.70)),
        gnn_val_ratio=float(gnn.get("val_ratio", 0.15)),
        min_cluster_size=int(camp.get("min_cluster_size", 10)),
        min_vehicles=int(config.get("campaign", {}).get("minimum_unique_vehicles", 2)),
        min_behavioral_cohesion=float(camp.get("min_behavioral_cohesion", 0.85)),
        dbscan_eps=float(camp.get("dbscan_eps", 0.8)),
        dbscan_min_samples=int(camp.get("dbscan_min_samples", 10)),
        dbscan_pca_components=int(camp.get("dbscan_pca_components", 8)),
        gnn_supervision=gnn.get("supervision", "structure"),
        seed=seed,
        retrain_gnn=True,
        checkpoint_path=None,
    )


def build_scenario_graph(
    descriptors: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    *,
    similarity_threshold: float | None = None,
    max_neighbors: int | None = None,
) -> GraphBuildResult:
    graph_cfg = config.get("graph", {})
    tau = similarity_threshold or float(graph_cfg.get("default_similarity_threshold", 0.95))
    k_same = int(graph_cfg.get("top_k_same_vehicle", 10))
    k_cross = int(graph_cfg.get("top_k_cross_vehicle", 5))
    if max_neighbors is not None:
        k_same = min(k_same, max_neighbors)
        k_cross = min(k_cross, max(1, max_neighbors // 2))

    t0 = time.perf_counter()
    scaler = resolve_fleet_scaler_from_config(config)
    X_gnn, feat_df, _ = prepare_gnn_fleet_node_matrix(
        descriptors, fleet_scaler_provenance=scaler
    )
    veh_col = resolve_vehicle_id_column(feat_df)
    vehicles = feat_df[veh_col].to_numpy()
    from src.graph.fleet_graph_builder import build_cross_vehicle_constrained_knn_edges

    _, _, edge_index, edge_weights, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X_gnn,
        vehicles,
        top_k_same_vehicle=k_same,
        top_k_cross_vehicle=k_cross,
        similarity_threshold=tau,
        metric="cosine",
        seed=seed,
    )
    meta = feat_df.iloc[sub_idx].reset_index(drop=True)
    X_sub = X_gnn[sub_idx]
    graph = build_networkx_graph(meta, edge_index, edge_weights)
    pyg = build_pyg_data(X_sub, edge_index, edge_weights, meta, prefer_ground_truth_labels=False)

    X_beh, _ = resolve_fleet_similarity_matrix(
        meta,
        similarity_feature_view=graph_cfg.get(
            "similarity_feature_view", "behavior_only_vehicle_normalized"
        ),
        feature_dominance_threshold=5.0,
        allowed_high_dominance_features=frozenset(),
        fleet_scaler_provenance=scaler,
    )
    nodes_df, edges_df = graph_to_tables(graph)
    stats = _extended_graph_stats(graph, edges_df, tau, k_same, k_cross, time.perf_counter() - t0)
    return GraphBuildResult(
        graph=graph,
        pyg_data=pyg,
        stats=stats,
        behavior_features=X_beh,
        meta=meta,
        edge_list=edges_df,
        graph_build_sec=time.perf_counter() - t0,
    )


def _extended_graph_stats(
    graph: nx.Graph,
    edges_df: pd.DataFrame,
    tau: float,
    k_same: int,
    k_cross: int,
    build_sec: float,
) -> pd.DataFrame:
    n = graph.number_of_nodes()
    m_unique = graph.number_of_edges()
    pyg_edges = m_unique * 2
    degrees = [d for _, d in graph.degree()]
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    med_deg = float(np.median(degrees)) if degrees else 0.0
    density = (2.0 * m_unique / (n * (n - 1))) if n > 1 else 0.0
    isolated = sum(1 for d in degrees if d == 0)
    cross = int(edges_df["is_cross_vehicle_edge"].sum()) if "is_cross_vehicle_edge" in edges_df.columns else 0
    return pd.DataFrame(
        [
            {
                "nodes": n,
                "unique_undirected_edges": m_unique,
                "pyg_stored_edges": pyg_edges,
                "directed_status": "undirected_nx_bidirectional_pyg",
                "average_degree": round(avg_deg, 4),
                "median_degree": round(med_deg, 4),
                "graph_density": round(density, 8),
                "isolated_nodes": isolated,
                "isolated_node_percentage": round(100.0 * isolated / max(n, 1), 4),
                "connected_components": nx.number_connected_components(graph),
                "largest_component_size": len(max(nx.connected_components(graph), key=len)) if n else 0,
                "cross_vehicle_edges": cross,
                "same_vehicle_edges": m_unique - cross,
                "cross_vehicle_edge_percentage": round(100.0 * cross / max(m_unique, 1), 4),
                "similarity_threshold": tau,
                "top_k_same_vehicle": k_same,
                "top_k_cross_vehicle": k_cross,
                "graph_construction_time_sec": round(build_sec, 6),
            }
        ]
    )


def _qualify_clusters_ieee(
    labels: np.ndarray,
    meta: pd.DataFrame,
    behavior_features: np.ndarray,
    cfg: FinalGnnFleetConfig,
    scores: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid in sorted({int(c) for c in np.unique(labels)}):
        if cid == -1:
            continue
        mask = labels == cid
        size = int(mask.sum())
        veh_col = resolve_vehicle_id_column(meta)
        n_veh = int(meta.loc[mask, veh_col].nunique())
        cohesion = compute_cluster_behavioral_cohesion(behavior_features, mask, seed=cfg.seed + cid)
        mean_score = float(scores[mask].mean()) if scores is not None else float(meta.loc[mask, "anomaly_score"].mean())
        dom = str(meta.loc[mask, "attack_type"].mode().iloc[0]) if size else "unknown"
        dom_ratio = float((meta.loc[mask, "attack_type"] == dom).mean()) if size else 0.0
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "behavioral_cohesion": round(cohesion, 4),
                "mean_anomaly_score": round(mean_score, 4),
                "eval_dominant_attack_type": dom,
                "eval_attack_type_purity": round(dom_ratio, 4),
                "is_qualifying_campaign_cluster": bool(
                    size >= cfg.min_cluster_size
                    and n_veh >= cfg.min_vehicles
                    and cohesion >= cfg.min_behavioral_cohesion
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_features(
    features: np.ndarray,
    meta: pd.DataFrame,
    cfg: FinalGnnFleetConfig,
) -> np.ndarray:
    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        features[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    return extend_dbscan_labels(features, fit_labels, features[fit_idx], projector, eps=cfg.dbscan_eps)


def _decisions_to_predictions(
    decisions: pd.DataFrame,
    membership: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    mem_cols = [
        "event_id",
        "ground_truth_malicious",
        "ground_truth_campaign_id",
        "scenario_role",
    ]
    for col in ("vehicle_token", "scenario_vehicle_id", "vehicle_model"):
        if col in membership.columns:
            mem_cols.append(col)
    out = decisions.merge(membership[mem_cols].drop_duplicates(subset=["event_id"]), on="event_id", how="left")
    out["method"] = method
    out["predicted_malicious"] = (
        (out["local_alert"] == 1)
        | (out.get("weak_signal", 0) == 1)
        | (out["final_decision"] == DECISION_COORDINATED)
    ).astype(int)
    return out


class _GCNFleetCorrelator(nn.Module):
    def __init__(self, in_ch: int, hidden: int, emb: int, nclass: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, emb)
        self.classifier = nn.Linear(emb, nclass)
        self.campaign_scorer = nn.Sequential(nn.Linear(emb, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        return z, self.classifier(z), self.campaign_scorer(z).squeeze(-1)


def _train_gcn_structure(
    pyg_data: Any,
    cfg: FinalGnnFleetConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = pyg_data.x.to(dev)
    edge_index = pyg_data.edge_index.to(dev)
    nclass = max(int(pyg_data.y.max().item()) + 1 if pyg_data.y.numel() else 2, 2)
    model = _GCNFleetCorrelator(x.size(1), cfg.gnn_hidden_channels, cfg.gnn_embedding_dim, nclass).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gnn_learning_rate, weight_decay=cfg.gnn_weight_decay)
    anom = x[:, 0]
    camp_target = (anom - anom.min()) / (anom.max() - anom.min() + 1e-9)
    t0 = time.perf_counter()
    for _ in range(cfg.gnn_epochs):
        model.train()
        optimizer.zero_grad()
        z, _, camp = model(x, edge_index)
        z_norm = F.normalize(z, dim=-1)
        src, dst = edge_index
        link_loss = (1.0 - (z_norm[src] * z_norm[dst]).sum(dim=-1)).mean()
        loss = link_loss + 0.25 * F.mse_loss(camp, camp_target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z, _, camp = model(x, edge_index)
    return z.cpu().numpy(), camp.cpu().numpy(), time.perf_counter() - t0


def run_graph_method(
    descriptors: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    method: MethodId,
    *,
    similarity_threshold: float | None = None,
    max_neighbors: int | None = None,
) -> MethodOutputs:
    cfg = _fleet_config_from_experiment(config, seed)
    if config.get("quick_test_epochs"):
        cfg = FinalGnnFleetConfig(**{**cfg.__dict__, "gnn_epochs": int(config["quick_test_epochs"])})

    gbuild = build_scenario_graph(
        descriptors, config, seed,
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors,
    )
    meta = gbuild.meta
    runtime: dict[str, float] = {"graph_construction_sec": gbuild.graph_build_sec}

    scaler = resolve_fleet_scaler_from_config(config)
    if method == "descriptor_clustering":
        X, _ = resolve_fleet_similarity_matrix(
            meta,
            similarity_feature_view=config.get("graph", {}).get(
                "similarity_feature_view", "behavior_only_vehicle_normalized"
            ),
            feature_dominance_threshold=5.0,
            allowed_high_dominance_features=frozenset(),
            fleet_scaler_provenance=scaler,
        )
        t0 = time.perf_counter()
        labels = _cluster_features(X, meta, cfg)
        runtime["clustering_sec"] = time.perf_counter() - t0
        scores = meta["anomaly_score"].to_numpy(dtype=np.float64)
        cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, scores)
        campaign_scores = scores
        embeddings = None
        gnn_sec = 0.0
    else:
        t0 = time.perf_counter()
        if method == "fcgnn":
            emb, campaign_scores, _ = run_gnn_fleet_correlation(
                gbuild.pyg_data, meta["event_id"].astype(str).tolist(), cfg
            )
        else:
            emb, campaign_scores, gnn_sec = _train_gcn_structure(gbuild.pyg_data, cfg)
        runtime["gnn_inference_sec"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        labels = _cluster_features(emb, meta, cfg)
        cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, campaign_scores)
        runtime["clustering_sec"] = time.perf_counter() - t1
        embeddings = emb

    if "weak_signal" not in meta.columns:
        meta = meta.copy()
        meta["weak_signal"] = 0
    decisions = assign_final_decisions(meta, labels, cluster_df, campaign_scores, cfg)
    decisions = decisions.merge(meta[["event_id", "weak_signal"]], on="event_id", how="left")
    event_pred = _decisions_to_predictions(decisions, membership, method)
    if "scenario_gt_malicious" in meta.columns:
        gt_map = meta.set_index("event_id")["scenario_gt_malicious"]
        event_pred["ground_truth_malicious"] = event_pred["event_id"].map(gt_map).fillna(
            event_pred["ground_truth_malicious"]
        ).astype(int)

    veh_col = resolve_vehicle_id_column(event_pred)
    vehicle_pred = (
        event_pred.groupby(veh_col, as_index=False)
        .agg(
            predicted_attacked=("predicted_malicious", "max"),
            ground_truth_attacked=("ground_truth_malicious", "max"),
            n_events=("event_id", "count"),
            n_coordinated=("final_decision", lambda s: int((s == DECISION_COORDINATED).sum())),
        )
        .rename(columns={veh_col: "vehicle_id"})
    )
    if "vehicle_model" in event_pred.columns:
        model_map = event_pred.groupby(veh_col)["vehicle_model"].first()
        vehicle_pred["vehicle_model"] = vehicle_pred["vehicle_id"].map(model_map)
    qualifying = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else cluster_df
    campaign_pred = pd.DataFrame(
        [
            {
                "method": method,
                "n_qualifying_campaign_clusters": len(qualifying),
                "n_detected_coordinated_events": int((event_pred["final_decision"] == DECISION_COORDINATED).sum()),
                "false_campaign_clusters": max(len(qualifying) - int(membership["ground_truth_campaign_id"].nunique() > 1), 0),
            }
        ]
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
