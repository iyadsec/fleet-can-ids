"""Fleet graph construction from behavioural similarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.ctt.constants import (
    GRAPH_CROSS_VEHICLE_CAP,
    GRAPH_KNN_CAP,
    GRAPH_SIMILARITY_THRESHOLD,
    OUTPUT_ROOT,
)
from src.ctt.descriptors import load_descriptor_vectors
from src.ctt.utils import ensure_dir, safe_div


def build_behavioural_graph(
    desc_df: pd.DataFrame,
    similarity_threshold: float = GRAPH_SIMILARITY_THRESHOLD,
    knn_cap: int = GRAPH_KNN_CAP,
    cross_vehicle_cap: int = GRAPH_CROSS_VEHICLE_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build fleet graph with behavioural similarity edges only (no temporal edges)."""
    if desc_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"num_nodes": 0, "num_edges": 0}

    X, event_ids = load_descriptor_vectors(desc_df)
    X = np.nan_to_num(X, nan=0.0)
    X_norm = StandardScaler().fit_transform(X)

    n = len(X)
    nn = NearestNeighbors(metric="cosine", n_neighbors=min(knn_cap + 1, n), n_jobs=-1)
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    id_to_meta = desc_df.set_index("event_id").to_dict("index")
    edges: list[dict] = []

    for i in range(n):
        cross_count: dict[str, int] = {}
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = 1.0 - float(dist)
            if sim < similarity_threshold:
                continue
            u, v = event_ids[i], event_ids[j]
            if u >= v:
                continue
            u_vid = id_to_meta[u].get("vehicle_id", "")
            v_vid = id_to_meta[v].get("vehicle_id", "")
            cross_vehicle = u_vid != v_vid
            if cross_vehicle:
                key = f"{u_vid}|{v_vid}"
                cross_count[key] = cross_count.get(key, 0) + 1
                if cross_count[key] > cross_vehicle_cap:
                    continue
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "similarity": sim,
                    "edge_type": "behavioural_similarity",
                    "cross_vehicle": cross_vehicle,
                    "cross_manufacturer": id_to_meta[u].get("manufacturer") != id_to_meta[v].get("manufacturer"),
                    "temporal_edge": False,
                }
            )

    edge_df = pd.DataFrame(edges).drop_duplicates(subset=["source", "target"]) if edges else pd.DataFrame()
    node_df = desc_df[["event_id", "vehicle_id", "manufacturer", "attack_type", "anomaly_score", "label"]].copy()
    node_df = node_df.rename(columns={"event_id": "node_id"})

    n_edges = len(edge_df)
    n_nodes = len(node_df)
    cross_v_edges = int(edge_df["cross_vehicle"].sum()) if n_edges else 0
    degrees = edge_df.groupby("source").size().add(edge_df.groupby("target").size(), fill_value=0) if n_edges else pd.Series(dtype=float)

    stats = {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "cross_vehicle_edges": cross_v_edges,
        "cross_vehicle_edge_pct": safe_div(cross_v_edges, n_edges) * 100,
        "cross_manufacturer_edges": int(edge_df["cross_manufacturer"].sum()) if n_edges else 0,
        "average_degree": float(degrees.mean()) if len(degrees) else 0.0,
        "isolated_node_rate": safe_div(n_nodes - len(degrees), n_nodes),
        "similarity_threshold": similarity_threshold,
        "knn_cap": knn_cap,
        "temporal_edges": 0,
        "mean_similarity": float(edge_df["similarity"].mean()) if n_edges else 0.0,
    }

    # Connected components
    if n_edges:
        import networkx as nx
        G = nx.Graph()
        for nid in node_df["node_id"]:
            G.add_node(nid)
        for _, e in edge_df.iterrows():
            G.add_edge(e["source"], e["target"])
        comps = list(nx.connected_components(G))
        stats["connected_components"] = len(comps)
        stats["largest_component"] = max(len(c) for c in comps) if comps else 0
    else:
        stats["connected_components"] = n_nodes
        stats["largest_component"] = 1 if n_nodes else 0

    return node_df, edge_df, stats


def save_graph_artifacts(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    stats: dict,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    graph_dir = ensure_dir(output_root / "graph")
    results_dir = ensure_dir(output_root / "results" / "graph_analysis")
    node_df.to_csv(graph_dir / "node_manifest.csv", index=False)
    edge_df.to_csv(graph_dir / "edge_list.csv", index=False)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(graph_dir / "graph_statistics.csv", index=False)
    stats_df.to_csv(results_dir / "graph_statistics.csv", index=False)
