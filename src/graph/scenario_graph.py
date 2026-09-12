"""Publication scenario graph construction (authoritative constrained-kNN path).

Restored from the campaign-clustering publication pipeline's graph stage.

The original ``build_scenario_graph`` in ``src/experiments/experiment_pipeline.py``
prepared GNN node features via ``prepare_gnn_fleet_node_matrix`` (depends on
``final_gnn_fleet_decision_experiment``, fleet scaler provenance, and related
modules that are not present on ``main``).

This module restores the **graph construction contract** used by that caller:

* cosine similarity
* separate same-vehicle / cross-vehicle neighbor caps
* similarity threshold pruning
* NetworkX + edge tables with vehicle identity metadata

Feature matrix construction uses ``resolve_fleet_similarity_matrix`` from the
authoritative ``fleet_graph_builder`` (same module that owns the constrained-kNN
edge builder). The unrecovered GNN feature-prep dependency is reported, not
re-invented.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .fleet_graph_builder import (
    build_cross_vehicle_constrained_knn_edges,
    build_networkx_graph,
    build_pyg_data,
    graph_to_tables,
    resolve_fleet_similarity_matrix,
)

logger = get_logger(__name__)

# Publication defaults (balanced OCSLab archive / audit JSON).
DEFAULT_SIMILARITY_THRESHOLD = 0.95
DEFAULT_MAX_SAME_VEHICLE_NEIGHBORS = 2
DEFAULT_MAX_CROSS_VEHICLE_NEIGHBORS = 5

# Modules required by the original experiment_pipeline GNN feature-prep path.
UNRECOVERED_GNN_FEATURE_PREP_DEPENDENCIES = (
    "src.evaluation.final_gnn_fleet_decision_experiment.prepare_gnn_fleet_node_matrix",
    "src.experiments.fleet_scaler_loader.resolve_fleet_scaler_from_config",
    "src.experiments.local_descriptor_normalisation",
    "src.experiments.final_shared_configuration (missing on all inspected branches)",
)


@dataclass(frozen=True)
class ScenarioGraphConfig:
    """Resolved publication graph parameters."""

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    max_same_vehicle_neighbors: int = DEFAULT_MAX_SAME_VEHICLE_NEIGHBORS
    max_cross_vehicle_neighbors: int = DEFAULT_MAX_CROSS_VEHICLE_NEIGHBORS
    similarity_feature_view: str = "full_descriptor"  # 24-D behavioural columns
    seed: int = 42


@dataclass
class GraphBuildResult:
    """Result of publication scenario graph construction."""

    graph: nx.Graph
    pyg_data: Any | None
    stats: pd.DataFrame
    behavior_features: np.ndarray
    meta: pd.DataFrame
    edge_list: pd.DataFrame
    graph_build_sec: float
    config: ScenarioGraphConfig


def resolve_publication_graph_config(
    config: dict[str, Any] | None = None,
    *,
    similarity_threshold: float | None = None,
    max_same_vehicle_neighbors: int | None = None,
    max_cross_vehicle_neighbors: int | None = None,
    seed: int | None = None,
) -> ScenarioGraphConfig:
    """
    Resolve graph parameters using publication names.

    Accepts both publication keys and campaign-clustering aliases:

    * ``similarity_threshold`` (alias: ``default_similarity_threshold``)
    * ``max_same_vehicle_neighbors`` (alias: ``top_k_same_vehicle``)
    * ``max_cross_vehicle_neighbors`` (alias: ``top_k_cross_vehicle``)
    """
    graph_cfg = (config or {}).get("graph", {}) if config else {}
    project_seed = (config or {}).get("project", {}).get("seed", 42) if config else 42

    tau = (
        similarity_threshold
        if similarity_threshold is not None
        else float(
            graph_cfg.get(
                "similarity_threshold",
                graph_cfg.get("default_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD),
            )
        )
    )
    k_same = (
        max_same_vehicle_neighbors
        if max_same_vehicle_neighbors is not None
        else int(
            graph_cfg.get(
                "max_same_vehicle_neighbors",
                graph_cfg.get("top_k_same_vehicle", DEFAULT_MAX_SAME_VEHICLE_NEIGHBORS),
            )
        )
    )
    k_cross = (
        max_cross_vehicle_neighbors
        if max_cross_vehicle_neighbors is not None
        else int(
            graph_cfg.get(
                "max_cross_vehicle_neighbors",
                graph_cfg.get("top_k_cross_vehicle", DEFAULT_MAX_CROSS_VEHICLE_NEIGHBORS),
            )
        )
    )
    view = str(graph_cfg.get("similarity_feature_view", "full_descriptor"))
    resolved_seed = int(seed if seed is not None else project_seed)
    return ScenarioGraphConfig(
        similarity_threshold=tau,
        max_same_vehicle_neighbors=k_same,
        max_cross_vehicle_neighbors=k_cross,
        similarity_feature_view=view,
        seed=resolved_seed,
    )


def _resolve_vehicle_column(meta: pd.DataFrame) -> str:
    for col in ("vehicle_token", "scenario_vehicle_id", "vehicle_id", "vehicle_model"):
        if col in meta.columns and meta[col].notna().any():
            return col
    raise ValueError(
        "Descriptors require a vehicle identity column "
        "(vehicle_token, scenario_vehicle_id, vehicle_id, or vehicle_model)."
    )


def _extended_graph_stats(
    graph: nx.Graph,
    edges_df: pd.DataFrame,
    cfg: ScenarioGraphConfig,
    build_sec: float,
) -> pd.DataFrame:
    n = graph.number_of_nodes()
    m_unique = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    avg_deg = float(np.mean(degrees)) if degrees else 0.0
    med_deg = float(np.median(degrees)) if degrees else 0.0
    density = (2.0 * m_unique / (n * (n - 1))) if n > 1 else 0.0
    isolated = sum(1 for d in degrees if d == 0)

    if not edges_df.empty and "is_cross_vehicle_edge" in edges_df.columns:
        cross = int(edges_df["is_cross_vehicle_edge"].astype(bool).sum())
    else:
        cross = 0
        for u, v in graph.edges():
            if graph.nodes[u].get("vehicle_model") != graph.nodes[v].get("vehicle_model"):
                cross += 1
    same = m_unique - cross
    return pd.DataFrame(
        [
            {
                "nodes": n,
                "unique_undirected_edges": m_unique,
                "pyg_stored_edges": m_unique * 2,
                "average_degree": round(avg_deg, 4),
                "median_degree": round(med_deg, 4),
                "graph_density": round(density, 8),
                "isolated_nodes": isolated,
                "isolated_node_percentage": round(100.0 * isolated / max(n, 1), 4),
                "connected_components": nx.number_connected_components(graph) if n else 0,
                "largest_component_size": (
                    len(max(nx.connected_components(graph), key=len)) if n else 0
                ),
                "cross_vehicle_edges": cross,
                "same_vehicle_edges": same,
                "cross_vehicle_edge_percentage": round(100.0 * cross / max(m_unique, 1), 4),
                "similarity_threshold": cfg.similarity_threshold,
                "max_same_vehicle_neighbors": cfg.max_same_vehicle_neighbors,
                "max_cross_vehicle_neighbors": cfg.max_cross_vehicle_neighbors,
                "similarity_metric": "cosine",
                "graph_construction_time_sec": round(build_sec, 6),
            }
        ]
    )


def build_scenario_graph_from_features(
    X: np.ndarray,
    vehicles: np.ndarray,
    meta: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
    max_same_vehicle_neighbors: int | None = None,
    max_cross_vehicle_neighbors: int | None = None,
    seed: int | None = None,
    build_pyg: bool = False,
) -> GraphBuildResult:
    """Build publication graph from an explicit feature matrix + vehicle IDs."""
    cfg = resolve_publication_graph_config(
        config,
        similarity_threshold=similarity_threshold,
        max_same_vehicle_neighbors=max_same_vehicle_neighbors,
        max_cross_vehicle_neighbors=max_cross_vehicle_neighbors,
        seed=seed,
    )
    if len(X) != len(vehicles) or len(X) != len(meta):
        raise ValueError("X, vehicles, and meta must have the same length")

    t0 = time.perf_counter()
    _, _, edge_index, edge_weights, sub_idx = build_cross_vehicle_constrained_knn_edges(
        np.asarray(X, dtype=np.float32),
        np.asarray(vehicles),
        top_k_same_vehicle=cfg.max_same_vehicle_neighbors,
        top_k_cross_vehicle=cfg.max_cross_vehicle_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        metric="cosine",
        seed=cfg.seed,
    )
    meta_sub = meta.iloc[sub_idx].reset_index(drop=True)
    X_sub = np.asarray(X, dtype=np.float32)[sub_idx]
    graph = build_networkx_graph(meta_sub, edge_index, edge_weights)
    _nodes_df, edges_df = graph_to_tables(graph)
    if not edges_df.empty and "is_cross_vehicle_edge" not in edges_df.columns:
        if {"source_vehicle", "target_vehicle"}.issubset(edges_df.columns):
            edges_df = edges_df.copy()
            edges_df["is_cross_vehicle_edge"] = (
                edges_df["source_vehicle"].astype(str) != edges_df["target_vehicle"].astype(str)
            )
    pyg = None
    if build_pyg:
        try:
            pyg = build_pyg_data(
                X_sub,
                edge_index,
                edge_weights,
                meta_sub,
                prefer_ground_truth_labels=False,
            )
        except Exception as exc:  # pragma: no cover - optional torch_geometric
            logger.warning("PyG build skipped: %s", exc)
    build_sec = time.perf_counter() - t0
    stats = _extended_graph_stats(graph, edges_df, cfg, build_sec)
    return GraphBuildResult(
        graph=graph,
        pyg_data=pyg,
        stats=stats,
        behavior_features=X_sub,
        meta=meta_sub,
        edge_list=edges_df,
        graph_build_sec=build_sec,
        config=cfg,
    )


def build_scenario_graph(
    descriptors: pd.DataFrame,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    *,
    similarity_threshold: float | None = None,
    max_same_vehicle_neighbors: int | None = None,
    max_cross_vehicle_neighbors: int | None = None,
    build_pyg: bool = False,
) -> GraphBuildResult:
    """
    Build the publication fleet scenario graph.

    Parameters
    ----------
    descriptors:
        Suspicious behavioural descriptors (one row per node). Must include a
        vehicle identity column and feature columns usable by
        ``resolve_fleet_similarity_matrix``.
    config:
        Experiment config dict; reads ``config['graph']`` publication keys.
    """
    cfg = resolve_publication_graph_config(
        config,
        similarity_threshold=similarity_threshold,
        max_same_vehicle_neighbors=max_same_vehicle_neighbors,
        max_cross_vehicle_neighbors=max_cross_vehicle_neighbors,
        seed=seed,
    )
    logger.info(
        "build_scenario_graph: threshold=%.3f same_k=%d cross_k=%d view=%s "
        "(GNN feature-prep path not restored: %s)",
        cfg.similarity_threshold,
        cfg.max_same_vehicle_neighbors,
        cfg.max_cross_vehicle_neighbors,
        cfg.similarity_feature_view,
        ", ".join(UNRECOVERED_GNN_FEATURE_PREP_DEPENDENCIES[:2]) + ", ...",
    )

    X, _cols = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=cfg.similarity_feature_view,  # type: ignore[arg-type]
    )
    veh_col = _resolve_vehicle_column(descriptors)
    vehicles = descriptors[veh_col].to_numpy()
    meta = descriptors.copy()
    if "vehicle_model" not in meta.columns:
        meta["vehicle_model"] = vehicles
    if "event_id" not in meta.columns:
        meta["event_id"] = [f"evt_{i}" for i in range(len(meta))]
    # Columns required by authoritative build_networkx_graph node attributes.
    defaults = {
        "window_id": -1,
        "source_file": "",
        "attack_type": "unknown",
        "anomaly_score": 0.0,
        "evidence_level": "weak",
        "local_alert": 0,
        "weak_signal": 1,
    }
    for col, default in defaults.items():
        if col not in meta.columns:
            meta[col] = default

    return build_scenario_graph_from_features(
        X,
        vehicles,
        meta,
        config=config,
        similarity_threshold=cfg.similarity_threshold,
        max_same_vehicle_neighbors=cfg.max_same_vehicle_neighbors,
        max_cross_vehicle_neighbors=cfg.max_cross_vehicle_neighbors,
        seed=cfg.seed,
        build_pyg=build_pyg,
    )
