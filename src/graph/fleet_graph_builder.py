"""Fleet-level anomaly graph from behavioural similarity between descriptors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_similarity_features import (
    SimilarityFeatureView,
    prepare_fleet_similarity_matrix,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

SimilarityMetric = Literal["cosine", "euclidean"]

DESCRIPTOR_PATH_COLUMNS = [
    "event_id",
    "window_id",
    "vehicle_model",
    "source_file",
    "attack_type",
    "anomaly_score",
    "evidence_level",
    "local_alert",
    "weak_signal",
]

OPTIONAL_DESCRIPTOR_COLUMNS = ("ground_truth_label",)


def event_id_to_window_id(event_id: str) -> int:
    return int(str(event_id).rsplit("-", 1)[-1])


def resolve_gnn_node_labels(df: pd.DataFrame, *, prefer_ground_truth: bool = True) -> np.ndarray:
    """
    Node classification targets for GNN training.

    Prefer dataset ``ground_truth_label`` when available so learning is not tied
    to IDS predictions. Falls back to ``predicted_label`` for legacy CSVs.
    """
    if prefer_ground_truth and "ground_truth_label" in df.columns:
        labels = df["ground_truth_label"].fillna(df["local_alert"]).astype(int).to_numpy()
        logger.info("GNN node labels: using ground_truth_label from descriptors")
    else:
        labels = (df["local_alert"].astype(int) | df["weak_signal"].astype(int)).to_numpy()
        logger.warning(
            "GNN node labels: ground_truth_label missing; using IDS evidence labels"
        )
    return labels


def attach_ground_truth_labels(
    descriptors: pd.DataFrame,
    features_path: Path | str | None,
) -> pd.DataFrame:
    """Merge window ``label`` into descriptors when ``ground_truth_label`` is absent."""
    if "ground_truth_label" in descriptors.columns:
        return descriptors
    if features_path is None or not Path(features_path).exists():
        return descriptors

    feat = pd.read_csv(
        Path(features_path),
        usecols=["window_id", "vehicle_model", "label"],
    )
    out = descriptors.copy()
    out["window_id"] = out["event_id"].map(event_id_to_window_id)
    out = out.merge(feat, on=["window_id", "vehicle_model"], how="left")
    out["ground_truth_label"] = out["label"].fillna(out["local_alert"]).astype(int)
    out = out.drop(columns=["label", "window_id"], errors="ignore")
    logger.info("Attached ground_truth_label from %s", features_path)
    return out


def load_anomaly_descriptors(
    path: Path | str,
    *,
    features_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load anomaly descriptor table (optionally enrich with ground-truth labels)."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Anomaly descriptors not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = set(DESCRIPTOR_PATH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Descriptor CSV missing columns: {sorted(missing)}")
    return attach_ground_truth_labels(df, features_path)


def parse_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return descriptor feature matrix from explicit columns or legacy JSON vectors."""
    node_ids = df["event_id"].astype(str).tolist()
    if set(BEHAVIOURAL_FEATURE_COLUMNS).issubset(df.columns):
        X = df[BEHAVIOURAL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    elif "behavioural_feature_vector" in df.columns:
        vectors: list[list[float]] = []
        for _, row in df.iterrows():
            raw = row["behavioural_feature_vector"]
            vals = json.loads(raw) if isinstance(raw, str) else raw
            vec = [float(v) if v is not None else 0.0 for v in vals]
            if len(vec) != len(BEHAVIOURAL_FEATURE_COLUMNS):
                raise ValueError(
                    f"Expected {len(BEHAVIOURAL_FEATURE_COLUMNS)} features, got {len(vec)} "
                    f"for {row['event_id']}"
                )
            vectors.append(vec)
        X = np.asarray(vectors, dtype=np.float32)
    else:
        missing = sorted(set(BEHAVIOURAL_FEATURE_COLUMNS) - set(df.columns))
        raise ValueError(f"Descriptor CSV missing feature columns: {missing}")
    X = np.nan_to_num(X, nan=0.0)
    return X, node_ids


def resolve_fleet_similarity_matrix(
    descriptors: pd.DataFrame,
    *,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
    fleet_scaler_provenance: Any | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for fleet graph similarity only (not IDS / stored descriptors)."""
    X, columns, _, _ = prepare_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=similarity_feature_view,
        feature_dominance_threshold=feature_dominance_threshold,
        allowed_high_dominance_features=allowed_high_dominance_features,
        fleet_scaler_provenance=fleet_scaler_provenance,
    )
    return X, columns


def _prepare_features(
    X: np.ndarray,
    metric: SimilarityMetric,
    *,
    scale_euclidean: bool = True,
) -> np.ndarray:
    """Scale features for euclidean similarity (cosine is scale-invariant)."""
    if metric == "euclidean" and scale_euclidean:
        return StandardScaler().fit_transform(X).astype(np.float32)
    return X


def _similarity_from_distance(distances: np.ndarray, metric: SimilarityMetric) -> np.ndarray:
    """Convert neighbour distances to similarity scores in [0, 1]."""
    if metric == "cosine":
        # sklearn cosine distance = 1 - cosine_similarity (for L2-normalized cosine)
        return np.clip(1.0 - distances, 0.0, 1.0)
    # Euclidean: sim = 1 / (1 + d)
    return 1.0 / (1.0 + distances)


def _directed_pairs_to_undirected(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse directed (u, v) pairs to one undirected edge per pair (max weight)."""
    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)
    edge_scores: dict[tuple[int, int], float] = {}
    for u, v, w in zip(src, dst, weights):
        if int(u) == int(v):
            continue
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        edge_scores[(a, b)] = max(float(w), edge_scores.get((a, b), 0.0))
    pairs = np.asarray(list(edge_scores.keys()), dtype=np.int64)
    w = np.asarray(list(edge_scores.values()), dtype=np.float32)
    edge_index = np.vstack([pairs[:, 0], pairs[:, 1]])
    return edge_index, w


def build_cross_vehicle_constrained_knn_edges(
    X: np.ndarray,
    vehicles: np.ndarray,
    *,
    top_k_same_vehicle: int = 10,
    top_k_cross_vehicle: int = 5,
    similarity_threshold: float = 0.95,
    metric: SimilarityMetric = "cosine",
    max_nodes: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cross-vehicle constrained kNN: per node, top-k same-vehicle + top-k cross-vehicle neighbours.

    Returns undirected edges before/after similarity threshold pruning.
    """
    if top_k_same_vehicle < 0 or top_k_cross_vehicle < 0:
        raise ValueError("top_k_same_vehicle and top_k_cross_vehicle must be >= 0")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in (0, 1], got {similarity_threshold}")
    if metric != "cosine":
        raise ValueError("build_cross_vehicle_constrained_knn_edges supports metric='cosine' only")

    X_work = X.copy()
    veh_work = np.asarray(vehicles)
    if max_nodes is not None and len(X_work) > max_nodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_work), size=max_nodes, replace=False)
        idx.sort()
        X_work = X_work[idx]
        veh_work = veh_work[idx]
        logger.info("Subsampled to %d nodes (max_nodes=%d)", len(X_work), max_nodes)
    else:
        idx = np.arange(len(X_work))

    n = len(X_work)
    if n < 2:
        empty_ei = np.zeros((2, 0), dtype=np.int64)
        empty_w = np.zeros(0, dtype=np.float32)
        return empty_ei, empty_w, empty_ei, empty_w, idx

    X_prep = _prepare_features(X_work, metric)
    src_list: list[int] = []
    dst_list: list[int] = []
    sim_list: list[float] = []

    unique_vehicles = pd.Series(veh_work).dropna().unique()
    for vehicle in unique_vehicles:
        source_idx = np.flatnonzero(veh_work == vehicle)
        if source_idx.size == 0:
            continue

        if top_k_same_vehicle > 0 and source_idx.size > 1:
            k_same = min(int(top_k_same_vehicle) + 1, source_idx.size)
            nn_same = NearestNeighbors(
                n_neighbors=k_same,
                metric="cosine",
                algorithm="brute" if source_idx.size <= 512 else "auto",
                n_jobs=1,
            )
            nn_same.fit(X_prep[source_idx])
            distances, local_idx = nn_same.kneighbors(X_prep[source_idx], return_distance=True)
            for row, i_global in enumerate(source_idx):
                for dist, j_local in zip(distances[row], local_idx[row]):
                    j_global = int(source_idx[int(j_local)])
                    if int(i_global) == j_global:
                        continue
                    sim = float(_similarity_from_distance(np.asarray([dist]), metric)[0])
                    src_list.append(int(i_global))
                    dst_list.append(j_global)
                    sim_list.append(sim)

        if top_k_cross_vehicle > 0:
            other_idx = np.flatnonzero(veh_work != vehicle)
            if other_idx.size == 0:
                continue
            k_cross = min(int(top_k_cross_vehicle), other_idx.size)
            nn_cross = NearestNeighbors(
                n_neighbors=k_cross,
                metric="cosine",
                algorithm="brute" if other_idx.size <= 512 else "auto",
                n_jobs=1,
            )
            nn_cross.fit(X_prep[other_idx])
            distances, local_idx = nn_cross.kneighbors(X_prep[source_idx], return_distance=True)
            for row, i_global in enumerate(source_idx):
                for dist, j_local in zip(distances[row], local_idx[row]):
                    j_global = int(other_idx[int(j_local)])
                    sim = float(_similarity_from_distance(np.asarray([dist]), metric)[0])
                    src_list.append(int(i_global))
                    dst_list.append(j_global)
                    sim_list.append(sim)

    if not src_list:
        empty_ei = np.zeros((2, 0), dtype=np.int64)
        empty_w = np.zeros(0, dtype=np.float32)
        return empty_ei, empty_w, empty_ei, empty_w, idx

    src = np.asarray(src_list, dtype=np.int64)
    dst = np.asarray(dst_list, dtype=np.int64)
    sim = np.asarray(sim_list, dtype=np.float32)

    edge_index_before, weights_before = _directed_pairs_to_undirected(src, dst, sim)
    keep = sim >= similarity_threshold
    edge_index_after, weights_after = _directed_pairs_to_undirected(
        src[keep], dst[keep], sim[keep]
    )

    logger.info(
        "Cross-vehicle kNN: k_same=%d, k_cross=%d, threshold=%.4f, n=%d, edges before=%d after=%d",
        top_k_same_vehicle,
        top_k_cross_vehicle,
        similarity_threshold,
        n,
        edge_index_before.shape[1] if edge_index_before.size else 0,
        edge_index_after.shape[1] if edge_index_after.size else 0,
    )
    return edge_index_before, weights_before, edge_index_after, weights_after, idx


def build_cross_vehicle_constrained_graph(
    descriptors: pd.DataFrame,
    *,
    top_k_same_vehicle: int = 10,
    top_k_cross_vehicle: int = 5,
    similarity_threshold: float = 0.95,
    max_nodes: int | None = None,
    seed: int = 42,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> tuple[nx.Graph, nx.Graph, dict[str, float], np.ndarray]:
    """Build fleet graph with same-vehicle and cross-vehicle kNN quotas."""
    X_full, _ = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=similarity_feature_view,
        feature_dominance_threshold=feature_dominance_threshold,
        allowed_high_dominance_features=allowed_high_dominance_features,
    )
    vehicles = descriptors["vehicle_model"].to_numpy()
    ei_before, w_before, ei_after, w_after, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X_full,
        vehicles,
        top_k_same_vehicle=top_k_same_vehicle,
        top_k_cross_vehicle=top_k_cross_vehicle,
        similarity_threshold=similarity_threshold,
        metric="cosine",
        max_nodes=max_nodes,
        seed=seed,
    )
    df_sub = descriptors.iloc[sub_idx].reset_index(drop=True)
    G_before = build_networkx_graph(df_sub, ei_before, w_before)
    G_after = build_networkx_graph(df_sub, ei_after, w_after)
    stats_before = compute_graph_statistics(G_before)
    stats_after = compute_graph_statistics(G_after)
    stats: dict[str, float] = {
        **{f"before_{k}": v for k, v in stats_before.items()},
        **{f"after_{k}": v for k, v in stats_after.items()},
        "top_k_same_vehicle": float(top_k_same_vehicle),
        "top_k_cross_vehicle": float(top_k_cross_vehicle),
        "similarity_threshold": similarity_threshold,
        "similarity_metric": "cosine",
        "similarity_feature_view": similarity_feature_view,
    }
    cross_clusters = _count_cross_vehicle_components(G_after)
    stats["cross_vehicle_cluster_count"] = float(cross_clusters)
    logger.info(
        "Cross-vehicle constrained graph: cross edges=%.0f, cross-vehicle clusters=%.0f",
        stats.get("after_num_cross_vehicle_edges", 0),
        cross_clusters,
    )
    return G_before, G_after, stats, sub_idx


def _count_cross_vehicle_components(G: nx.Graph) -> int:
    count = 0
    for component in nx.connected_components(G):
        vehicles = {G.nodes[n].get("vehicle_model") for n in component}
        if len(vehicles) >= 2:
            count += 1
    return count


def build_networkx_graph(
    df: pd.DataFrame,
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    node_index_map: np.ndarray | None = None,
) -> nx.Graph:
    """Build an undirected NetworkX graph (nodes = anomaly descriptors)."""
    if node_index_map is not None:
        df = df.iloc[node_index_map].reset_index(drop=True)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(
            row["event_id"],
            window_id=int(row["window_id"]) if "window_id" in row and pd.notna(row["window_id"]) else -1,
            vehicle_model=row["vehicle_model"],
            source_file=row["source_file"] if "source_file" in row else "",
            attack_type=row["attack_type"],
            anomaly_score=float(row["anomaly_score"]),
            evidence_level=row["evidence_level"],
            local_alert=int(row["local_alert"]),
            weak_signal=int(row["weak_signal"]),
        )

    id_list = df["event_id"].tolist()
    if edge_index.size > 0:
        for k in range(edge_index.shape[1]):
            u = id_list[int(edge_index[0, k])]
            v = id_list[int(edge_index[1, k])]
            w = float(edge_weights[k])
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], w)
            else:
                G.add_edge(u, v, weight=w, similarity=w)

    return G


def graph_to_tables(G: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the fleet graph to node and edge tables for transparent inspection."""
    nodes = pd.DataFrame(
        [{"event_id": node, **attrs} for node, attrs in G.nodes(data=True)]
    )
    edges = pd.DataFrame(
        [
            {
                "source_event_id": u,
                "target_event_id": v,
                "source_vehicle": G.nodes[u].get("vehicle_model", ""),
                "target_vehicle": G.nodes[v].get("vehicle_model", ""),
                "similarity_score": attrs.get("similarity", attrs.get("weight", 1.0)),
                "is_cross_vehicle_edge": G.nodes[u].get("vehicle_model", "")
                != G.nodes[v].get("vehicle_model", ""),
            }
            for u, v, attrs in G.edges(data=True)
        ]
    )
    return nodes, edges


def save_graph_tables(
    G: nx.Graph,
    *,
    nodes_path: Path | str,
    edges_path: Path | str,
) -> tuple[Path, Path]:
    nodes, edges = graph_to_tables(G)
    nodes_out = Path(nodes_path)
    edges_out = Path(edges_path)
    nodes_out.parent.mkdir(parents=True, exist_ok=True)
    edges_out.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(nodes_out, index=False)
    edges.to_csv(edges_out, index=False)
    logger.info("Saved fleet node table to %s", nodes_out)
    logger.info("Saved fleet edge table to %s", edges_out)
    return nodes_out, edges_out


def build_pyg_data(
    X: np.ndarray,
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    df: pd.DataFrame,
    *,
    prefer_ground_truth_labels: bool = True,
) -> Any:
    """Build a PyTorch Geometric ``Data`` object."""
    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch-geometric is required. Install with: pip install torch-geometric"
        ) from exc

    x = torch.tensor(X, dtype=torch.float32)
    ei = torch.tensor(edge_index, dtype=torch.long)
    ew = torch.tensor(edge_weights, dtype=torch.float32)

    from src.experiments.vehicle_identity import resolve_graph_vehicle_column

    veh_col = resolve_graph_vehicle_column(df)
    vehicle_codes = {v: i for i, v in enumerate(sorted(df[veh_col].astype(str).unique()))}
    attack_codes = {a: i for i, a in enumerate(sorted(df["attack_type"].unique()))}

    data = Data(
        x=x,
        edge_index=ei,
        edge_attr=ew,
        y=torch.tensor(
            resolve_gnn_node_labels(df, prefer_ground_truth=prefer_ground_truth_labels),
            dtype=torch.long,
        ),
        local_alert=torch.tensor(df["local_alert"].to_numpy(), dtype=torch.long),
        weak_signal=torch.tensor(df["weak_signal"].to_numpy(), dtype=torch.long),
        anomaly_score=torch.tensor(df["anomaly_score"].to_numpy(), dtype=torch.float32),
        vehicle_id=torch.tensor(
            [vehicle_codes[str(v)] for v in df[veh_col].astype(str)], dtype=torch.long
        ),
        attack_id=torch.tensor(
            [attack_codes[a] for a in df["attack_type"]], dtype=torch.long
        ),
    )
    data.num_nodes = x.size(0)
    data.event_ids = df["event_id"].tolist()
    data.vehicle_label_map = vehicle_codes
    data.vehicle_id_column = veh_col
    data.attack_label_map = attack_codes
    return data


def compute_graph_statistics(G: nx.Graph) -> dict[str, float]:
    """Compute fleet graph statistics (undirected)."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "average_degree": 0.0,
            "graph_density": 0.0,
            "num_cross_vehicle_edges": 0.0,
            "connected_components": 0.0,
        }
    avg_degree = float(2 * m / n)
    density = float(2 * m / (n * (n - 1))) if n > 1 else 0.0
    return {
        "num_nodes": float(n),
        "num_edges": float(m),
        "average_degree": avg_degree,
        "graph_density": density,
        "num_cross_vehicle_edges": float(
            sum(
                1
                for u, v in G.edges()
                if G.nodes[u].get("vehicle_model") != G.nodes[v].get("vehicle_model")
            )
        ),
        "connected_components": float(nx.number_connected_components(G)),
    }


def save_fleet_graph(
    G: nx.Graph,
    pyg_data: Any,
    stats: dict[str, float],
    *,
    pt_path: Path | str,
    graphml_path: Path | str,
) -> tuple[Path, Path]:
    """Persist PyG bundle (.pt) and NetworkX graph (.graphml)."""
    pt_out = Path(pt_path)
    graphml_out = Path(graphml_path)
    pt_out.parent.mkdir(parents=True, exist_ok=True)
    graphml_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "pyg_data": pyg_data,
            "stats": stats,
            "event_ids": getattr(pyg_data, "event_ids", []),
        },
        pt_out,
    )
    nx.write_graphml(G, graphml_out)
    logger.info("Saved PyG graph to %s", pt_out)
    logger.info("Saved NetworkX graph to %s", graphml_out)
    return pt_out, graphml_out


def print_graph_statistics(stats: dict[str, float]) -> None:
    """Print graph statistics to stdout."""
    print("\n=== Fleet Graph Statistics ===")
    print(f"Number of nodes:    {int(stats.get('num_nodes', 0)):,}")
    print(f"Number of edges:    {int(stats.get('num_edges', 0)):,}")
    print(f"Average degree:     {stats.get('average_degree', 0):.4f}")
    print(f"Graph density:      {stats.get('graph_density', 0):.6f}")
    if "similarity_metric" in stats:
        print(f"Similarity metric:  {stats['similarity_metric']}")
        print(f"Threshold:          {stats.get('similarity_threshold')}")
    if stats.get("max_same_vehicle_neighbors") is not None:
        print(f"Max same-vehicle k: {int(stats['max_same_vehicle_neighbors'])}")
    if stats.get("max_cross_vehicle_neighbors") is not None:
        print(f"Max cross-vehicle k:{int(stats['max_cross_vehicle_neighbors'])}")
    if "directed_edge_count" in stats:
        print(f"Directed edges (PyG): {int(stats['directed_edge_count']):,}")
    print("==============================\n")
