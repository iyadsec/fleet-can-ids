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
    "predicted_label",
    "is_anomaly",
    "behavioural_feature_vector",
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
        labels = df["ground_truth_label"].fillna(df["predicted_label"]).astype(int).to_numpy()
        logger.info("GNN node labels: using ground_truth_label from descriptors")
    else:
        labels = df["predicted_label"].astype(int).to_numpy()
        logger.warning(
            "GNN node labels: ground_truth_label missing; using predicted_label (IDS-derived)"
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
    out["ground_truth_label"] = out["label"].fillna(out["predicted_label"]).astype(int)
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
    """Parse JSON feature vectors into a float matrix."""
    vectors: list[list[float]] = []
    node_ids: list[str] = []
    for _, row in df.iterrows():
        raw = row["behavioural_feature_vector"]
        if isinstance(raw, str):
            vals = json.loads(raw)
        else:
            vals = raw
        vec = [float(v) if v is not None else 0.0 for v in vals]
        if len(vec) != len(BEHAVIOURAL_FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(BEHAVIOURAL_FEATURE_COLUMNS)} features, got {len(vec)} "
                f"for {row['event_id']}"
            )
        vectors.append(vec)
        node_ids.append(str(row["event_id"]))
    X = np.asarray(vectors, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X, node_ids


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


def build_similarity_edges(
    X: np.ndarray,
    *,
    metric: SimilarityMetric = "cosine",
    threshold: float = 0.85,
    max_nodes: int | None = None,
    max_neighbors: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build undirected edges where behavioural similarity >= *threshold*.

    Uses radius neighbour search (no temporal features).
    Returns edge_index [2, E], edge_weights [E], and the (possibly subsampled) feature matrix.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    X_work = X.copy()
    if max_nodes is not None and len(X_work) > max_nodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_work), size=max_nodes, replace=False)
        idx.sort()
        X_work = X_work[idx]
        logger.info("Subsampled to %d nodes (max_nodes=%d)", len(X_work), max_nodes)
    else:
        idx = np.arange(len(X_work))

    X_prep = _prepare_features(X_work, metric)

    if metric == "cosine":
        radius = max(1.0 - threshold, 1e-6)
        nn_metric = "cosine"
    else:
        # Convert similarity threshold to distance: 1/(1+d) >= t  =>  d <= (1/t) - 1
        radius = max((1.0 / threshold) - 1.0, 1e-6)
        nn_metric = "euclidean"

    logger.info(
        "Nearest neighbours: metric=%s, radius=%.4f, threshold=%.4f, n=%d, max_neighbors=%s",
        metric,
        radius,
        threshold,
        len(X_prep),
        max_neighbors,
    )

    if max_neighbors is not None and max_neighbors > 0:
        n_neighbors = min(int(max_neighbors) + 1, len(X_prep))
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=nn_metric,
            algorithm="auto",
            n_jobs=1,
        )
        nn.fit(X_prep)
        distances, indices = nn.kneighbors(X_prep, return_distance=True)
        row_idx = np.repeat(np.arange(len(X_prep), dtype=np.int64), n_neighbors)
        col_idx = indices.reshape(-1).astype(np.int64, copy=False)
        dist = distances.reshape(-1)
        upper = row_idx < col_idx
        src = row_idx[upper]
        dst = col_idx[upper]
        sim = _similarity_from_distance(dist[upper], metric).astype(np.float32, copy=False)
    else:
        nn = NearestNeighbors(
            radius=radius,
            metric=nn_metric,
            algorithm="auto",
            n_jobs=1,
        )
        nn.fit(X_prep)
        distances = nn.radius_neighbors_graph(X_prep, mode="distance")
        coo = distances.tocoo()

        # Keep one side of the undirected graph and drop self-neighbours without
        # materializing Python lists of neighbours for every node.
        upper = coo.row < coo.col
        src = coo.row[upper].astype(np.int64, copy=False)
        dst = coo.col[upper].astype(np.int64, copy=False)
        sim = _similarity_from_distance(coo.data[upper], metric).astype(np.float32, copy=False)
    keep = sim >= threshold
    src = src[keep]
    dst = dst[keep]
    w = sim[keep]

    if len(src) == 0:
        logger.warning("No edges above threshold %.4f", threshold)
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weights = np.zeros(0, dtype=np.float32)
    else:
        # Bidirectional for PyG
        edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        edge_weights = np.concatenate([w, w])

    return edge_index, edge_weights, idx


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
            predicted_label=int(row["predicted_label"]),
            is_anomaly=int(row["is_anomaly"]) if "is_anomaly" in row else int(row["predicted_label"]),
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
                "similarity": attrs.get("similarity", attrs.get("weight", 1.0)),
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

    vehicle_codes = {v: i for i, v in enumerate(sorted(df["vehicle_model"].unique()))}
    attack_codes = {a: i for i, a in enumerate(sorted(df["attack_type"].unique()))}

    data = Data(
        x=x,
        edge_index=ei,
        edge_attr=ew,
        y=torch.tensor(
            resolve_gnn_node_labels(df, prefer_ground_truth=prefer_ground_truth_labels),
            dtype=torch.long,
        ),
        predicted_label=torch.tensor(df["predicted_label"].to_numpy(), dtype=torch.long),
        anomaly_score=torch.tensor(df["anomaly_score"].to_numpy(), dtype=torch.float32),
        vehicle_id=torch.tensor(
            [vehicle_codes[v] for v in df["vehicle_model"]], dtype=torch.long
        ),
        attack_id=torch.tensor(
            [attack_codes[a] for a in df["attack_type"]], dtype=torch.long
        ),
    )
    data.num_nodes = x.size(0)
    data.event_ids = df["event_id"].tolist()
    data.vehicle_label_map = vehicle_codes
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
        }
    avg_degree = float(2 * m / n)
    density = float(2 * m / (n * (n - 1))) if n > 1 else 0.0
    return {
        "num_nodes": float(n),
        "num_edges": float(m),
        "average_degree": avg_degree,
        "graph_density": density,
    }


def build_fleet_anomaly_graph(
    descriptors: pd.DataFrame,
    *,
    metric: SimilarityMetric = "cosine",
    threshold: float = 0.85,
    max_nodes: int | None = None,
    max_neighbors: int | None = None,
    seed: int = 42,
    prefer_ground_truth_labels: bool = True,
) -> tuple[nx.Graph, Any, dict[str, float], np.ndarray]:
    """
    Full pipeline: descriptors -> NetworkX + PyG graphs + statistics.

    Returns (nx_graph, pyg_data, stats, node_subsample_index).
    """
    X_full, _ = parse_feature_matrix(descriptors)
    edge_index, edge_weights, sub_idx = build_similarity_edges(
        X_full,
        metric=metric,
        threshold=threshold,
        max_nodes=max_nodes,
        max_neighbors=max_neighbors,
        seed=seed,
    )

    df_sub = descriptors.iloc[sub_idx].reset_index(drop=True)
    X_sub = X_full[sub_idx]

    G = build_networkx_graph(df_sub, edge_index, edge_weights)
    pyg_data = build_pyg_data(
        X_sub,
        edge_index,
        edge_weights,
        df_sub,
        prefer_ground_truth_labels=prefer_ground_truth_labels,
    )
    stats = compute_graph_statistics(G)
    stats["similarity_metric"] = metric
    stats["similarity_threshold"] = threshold
    stats["max_neighbors"] = float(max_neighbors) if max_neighbors else 0.0
    stats["directed_edge_count"] = float(edge_index.shape[1]) if edge_index.size else 0.0

    logger.info(
        "Fleet graph: %d nodes, %d edges (avg degree=%.2f, density=%.6f)",
        int(stats["num_nodes"]),
        int(stats["num_edges"]),
        stats["average_degree"],
        stats["graph_density"],
    )
    return G, pyg_data, stats, sub_idx


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
    if stats.get("max_neighbors"):
        print(f"Max neighbours:     {int(stats['max_neighbors'])}")
    if "directed_edge_count" in stats:
        print(f"Directed edges (PyG): {int(stats['directed_edge_count']):,}")
    print("==============================\n")
