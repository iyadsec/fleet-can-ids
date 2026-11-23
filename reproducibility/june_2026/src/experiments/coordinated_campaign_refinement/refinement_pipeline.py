"""Refinement pipeline: candidate-only graph, controlled connectivity, consolidation."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import (
    DECISION_COORDINATED,
    assign_final_decisions,
    run_gnn_fleet_correlation,
)
from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
from src.experiments.experiment_pipeline import (
    GraphBuildResult,
    MethodOutputs,
    _cluster_features,
    _decisions_to_predictions,
    _fleet_config_from_experiment,
    _qualify_clusters_ieee,
    build_scenario_graph,
)
from src.experiments.fleet_scaler_loader import resolve_fleet_scaler_from_config
from src.experiments.vehicle_identity import resolve_vehicle_id_column
from src.graph.fleet_graph_builder import build_cross_vehicle_constrained_knn_edges, build_networkx_graph, build_pyg_data, graph_to_tables, resolve_fleet_similarity_matrix
from src.evaluation.final_gnn_fleet_decision_experiment import prepare_gnn_fleet_node_matrix


@dataclass(frozen=True)
class RefinementDesign:
    design_id: str
    candidate_only: bool
    similarity_threshold: float = 0.95
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 10
    consolidate_fragments: bool = False


def build_refinement_graph(
    descriptors: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    design: RefinementDesign,
) -> tuple[GraphBuildResult, dict[str, Any], pd.DataFrame]:
    """Build graph from full or candidate-only nodes."""
    ann = annotate_fleet_candidates(
        descriptors,
        weak_threshold=float(config.get("local_ids", {}).get("weak_threshold", 0.55)),
        strong_threshold=float(config.get("local_ids", {}).get("strong_threshold", 0.80)),
    )
    graph_input = ann[ann["fleet_candidate"]].copy() if design.candidate_only else ann.copy()
    cand_stats = candidate_statistics(ann)

    t0 = time.perf_counter()
    scaler = resolve_fleet_scaler_from_config(config)
    X_gnn, feat_df, _ = prepare_gnn_fleet_node_matrix(graph_input, fleet_scaler_provenance=scaler)
    veh_col = resolve_vehicle_id_column(feat_df)
    vehicles = feat_df[veh_col].to_numpy()

    _, _, edge_index, edge_weights, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X_gnn,
        vehicles,
        top_k_same_vehicle=design.top_k_same_vehicle,
        top_k_cross_vehicle=design.top_k_cross_vehicle,
        similarity_threshold=design.similarity_threshold,
        metric="cosine",
        seed=seed,
    )
    meta = feat_df.iloc[sub_idx].reset_index(drop=True)
    X_sub = X_gnn[sub_idx]
    graph = build_networkx_graph(meta, edge_index, edge_weights)
    pyg = build_pyg_data(X_sub, edge_index, edge_weights, meta, prefer_ground_truth_labels=False)

    graph_cfg = config.get("graph", {})
    X_beh, _ = resolve_fleet_similarity_matrix(
        meta,
        similarity_feature_view=graph_cfg.get("similarity_feature_view", "behavior_only_vehicle_normalized"),
        fleet_scaler_provenance=scaler,
    )
    nodes_df, edges_df = graph_to_tables(graph)
    build_sec = time.perf_counter() - t0

    n = graph.number_of_nodes()
    iso = sum(1 for _, d in graph.degree() if d == 0)
    cross_edges = 0
    same_edges = 0
    if not edges_df.empty and "source_vehicle" in edges_df.columns:
        cross_edges = int((edges_df["source_vehicle"] != edges_df["target_vehicle"]).sum())
        same_edges = len(edges_df) - cross_edges
    comp_sizes = [len(c) for c in nx.connected_components(graph)] if n else [0]

    stats = pd.DataFrame([{
        "nodes": n,
        "unique_undirected_edges": graph.number_of_edges(),
        "average_degree": float(np.mean([d for _, d in graph.degree()])) if n else 0,
        "graph_density": nx.density(graph) if n > 1 else 0,
        "isolated_nodes": iso,
        "isolated_node_percentage": 100.0 * iso / max(n, 1),
        "connected_components": nx.number_connected_components(graph),
        "largest_component_size": max(comp_sizes) if comp_sizes else 0,
        "cross_vehicle_edges": cross_edges,
        "same_vehicle_edges": same_edges,
        "cross_vehicle_edge_percentage": 100.0 * cross_edges / max(len(edges_df), 1),
        "similarity_threshold": design.similarity_threshold,
        "top_k_same_vehicle": design.top_k_same_vehicle,
        "top_k_cross_vehicle": design.top_k_cross_vehicle,
        "graph_construction_time_sec": build_sec,
    }])

    gbuild = GraphBuildResult(
        graph=graph,
        pyg_data=pyg,
        stats=stats,
        behavior_features=X_beh,
        meta=meta,
        edge_list=edges_df,
        graph_build_sec=build_sec,
    )
    return gbuild, cand_stats, ann


def _consolidate_fragment_labels(
    labels: np.ndarray,
    meta: pd.DataFrame,
    edge_list: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    min_vehicles: int = 2,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge qualifying clusters with cross-vehicle graph path and similar centroids."""
    decisions = []
    label_map = {int(c): int(c) for c in np.unique(labels) if int(c) >= 0}
    qual = [c for c in label_map if int((labels == c).sum()) >= 3]
    if len(qual) <= 1 or edge_list.empty:
        return labels, pd.DataFrame(decisions)

    veh_col = resolve_vehicle_id_column(meta)
    eid_to_cluster = dict(zip(meta["event_id"].astype(str), labels))
    G = nx.Graph()
    for _, e in edge_list.iterrows():
        if e["source_vehicle"] == e["target_vehicle"]:
            continue
        cs = eid_to_cluster.get(str(e["source_event_id"]))
        ct = eid_to_cluster.get(str(e["target_event_id"]))
        if cs is None or ct is None or cs < 0 or ct < 0:
            continue
        G.add_edge(int(cs), int(ct))

    centroids = {}
    for cid in qual:
        mask = labels == cid
        centroids[cid] = embeddings[mask].mean(axis=0)

    merged = set()
    for i, a in enumerate(qual):
        for b in qual[i + 1:]:
            if a in merged or b in merged:
                continue
            if a not in G or b not in G:
                continue
            if not nx.has_path(G, a, b):
                continue
            ca, cb = centroids[a], centroids[b]
            sim = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))
            if sim >= 0.85:
                label_map[b] = label_map[a]
                merged.add(b)
                decisions.append({"from_cluster": b, "to_cluster": a, "centroid_cosine": sim, "reason": "cross_vehicle_path"})

    new_labels = labels.copy()
    for old, new in label_map.items():
        new_labels[labels == old] = new
    return new_labels, pd.DataFrame(decisions)


def run_refinement_fcgnn(
    scenario_df: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    design: RefinementDesign,
) -> tuple[MethodOutputs, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = _fleet_config_from_experiment(config, seed)
    cfg = type(cfg)(**{**cfg.__dict__, "dbscan_eps": design.dbscan_eps, "dbscan_min_samples": design.dbscan_min_samples})

    tracemalloc.start()
    t0 = time.perf_counter()
    gbuild, cand_stats, annotated = build_refinement_graph(scenario_df, config, seed, design)
    meta = gbuild.meta
    runtime: dict[str, float] = {"graph_construction_sec": gbuild.graph_build_sec}

    t1 = time.perf_counter()
    emb, campaign_scores, _ = run_gnn_fleet_correlation(
        gbuild.pyg_data, meta["event_id"].astype(str).tolist(), cfg
    )
    runtime["gnn_inference_sec"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    labels = _cluster_features(emb, meta, cfg)
    consolidation_log = pd.DataFrame()
    if design.consolidate_fragments:
        labels, consolidation_log = _consolidate_fragment_labels(labels, meta, gbuild.edge_list, emb)
    cluster_df = _qualify_clusters_ieee(labels, meta, gbuild.behavior_features, cfg, campaign_scores)
    runtime["clustering_sec"] = time.perf_counter() - t2

    if "weak_signal" not in meta.columns:
        meta = meta.copy()
        meta["weak_signal"] = (meta.get("local_evidence_level", "") == "weak").astype(int)

    decisions = assign_final_decisions(meta, labels, cluster_df, campaign_scores, cfg)
    decisions = decisions.merge(meta[["event_id", "weak_signal"]], on="event_id", how="left")
    event_pred = _decisions_to_predictions(decisions, membership, "fcgnn")

    ann_map = annotated.set_index("event_id")
    event_pred["fleet_candidate"] = event_pred["event_id"].map(ann_map["fleet_candidate"]).fillna(False).astype(bool)
    event_pred["candidate_reason"] = event_pred["event_id"].map(ann_map["candidate_reason"]).fillna("not_in_scenario")
    event_pred["local_evidence_level"] = event_pred["event_id"].map(ann_map["local_evidence_level"]).fillna("unknown")
    event_pred["campaign_member_prediction"] = (event_pred["final_decision"] == DECISION_COORDINATED).astype(int)

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

    qualifying = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else cluster_df
    campaign_pred = pd.DataFrame([{
        "method": "fcgnn",
        "n_qualifying_campaign_clusters": len(qualifying),
        "n_detected_coordinated_events": int((event_pred["final_decision"] == DECISION_COORDINATED).sum()),
        "false_campaign_clusters": max(len(qualifying) - 1, 0),
    }])

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    runtime["total_sec"] = time.perf_counter() - t0
    runtime["peak_memory_mb"] = peak / (1024 * 1024)

    outputs = MethodOutputs(
        event_predictions=event_pred,
        vehicle_predictions=vehicle_pred,
        campaign_predictions=campaign_pred,
        cluster_df=cluster_df,
        embeddings=emb,
        graph_stats=gbuild.stats,
        edge_list=gbuild.edge_list,
        runtime=runtime,
    )
    return outputs, cand_stats, annotated, consolidation_log
