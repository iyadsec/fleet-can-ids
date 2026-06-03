"""Fleet-level correlation improvement experiment (local IDS vs fleet-aware IDS)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.graph.fleet_graph_builder import (
    build_fleet_correlation_graph,
    load_anomaly_descriptors,
    resolve_fleet_similarity_matrix,
)
from src.graph.fleet_similarity_features import SimilarityFeatureView
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

VEHICLE_COLORS = {
    "Chevrolet": "#4472C4",
    "Hyundai": "#ED7D31",
    "Kia": "#70AD47",
}


@dataclass(frozen=True)
class FleetCorrelationOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class FleetCorrelationConfig:
    top_k_neighbors: int = 15
    similarity_threshold: float = 0.95
    minimum_cluster_size: int = 2
    minimum_vehicle_count: int = 3
    fleet_cluster_score_threshold: float = 0.7
    max_graph_viz_nodes: int = 800
    graph_viz_seed: int = 42
    similarity_feature_view: SimilarityFeatureView = "full_descriptor"
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _df_to_ieee_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, float_format="%.4f")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            body.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": float(fpr),
        "detection_score": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _pct_gain(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("nan")
    return 100.0 * (new - old) / old


def load_graph_from_artifacts(
    nodes_path: Path,
    edges_path: Path,
) -> nx.Graph:
    """Build NetworkX graph from saved fleet node/edge tables."""
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    G = nx.Graph()
    id_col = "event_id" if "event_id" in nodes.columns else nodes.columns[0]
    for _, row in nodes.iterrows():
        nid = str(row[id_col])
        G.add_node(nid, **{k: row[k] for k in nodes.columns if k != id_col})
    for _, row in edges.iterrows():
        src = str(row["source_event_id"])
        tgt = str(row["target_event_id"])
        if src in G and tgt in G:
            G.add_edge(src, tgt, weight=float(row.get("similarity_score", 1.0)))
    return G


def compute_graph_statistics(G: nx.Graph) -> dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "number_of_nodes": 0,
            "number_of_edges": 0,
            "average_degree": 0.0,
            "graph_density": 0.0,
            "connected_components": 0,
            "largest_component_size": 0,
        }
    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    largest = max((len(c) for c in components), default=0)
    density = float(nx.density(G))
    return {
        "number_of_nodes": int(n),
        "number_of_edges": int(m),
        "average_degree": float(np.mean(degrees)),
        "graph_density": density,
        "connected_components": int(len(components)),
        "largest_component_size": int(largest),
    }


_DEGENERATE_GRAPH_FRACTION = 0.9


def identify_suspicious_behavioural_clusters(
    cluster_results: pd.DataFrame,
    *,
    minimum_cluster_size: int,
    minimum_vehicle_count: int,
    similarity_threshold: float,
    fleet_cluster_score_threshold: float,
) -> tuple[set[str], pd.DataFrame]:
    """Mark nodes in multi-vehicle behavioural clusters (DBSCAN / KMeans on descriptors)."""
    suspicious_nodes: set[str] = set()
    rows: list[dict[str, Any]] = []

    for cid, grp in cluster_results.groupby("cluster_id"):
        if int(cid) == -1:
            continue
        cluster_size = int(len(grp))
        if cluster_size < minimum_cluster_size:
            continue
        n_vehicles = int(grp["num_unique_vehicles"].iloc[0])
        mean_sim = float(grp["mean_cluster_similarity"].iloc[0])
        mean_score = float(grp["anomaly_score"].mean())
        is_suspicious = (
            cluster_size >= minimum_cluster_size
            and n_vehicles >= minimum_vehicle_count
            and mean_sim >= similarity_threshold
            and mean_score >= fleet_cluster_score_threshold
        )
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_method": "behavioural_clustering",
                "cluster_size": cluster_size,
                "num_unique_vehicles": n_vehicles,
                "mean_cluster_similarity": round(mean_sim, 6),
                "mean_anomaly_score": round(mean_score, 6),
                "is_fleet_suspicious_cluster": int(is_suspicious),
            }
        )
        if is_suspicious:
            suspicious_nodes.update(grp["event_id"].astype(str).tolist())

    return suspicious_nodes, pd.DataFrame(rows)


def identify_suspicious_connected_components(
    G: nx.Graph,
    descriptors: pd.DataFrame,
    *,
    minimum_cluster_size: int,
    minimum_vehicle_count: int,
    fleet_cluster_score_threshold: float,
) -> tuple[set[str], pd.DataFrame]:
    """Return event_ids in fleet-suspicious connected components."""
    desc = descriptors.set_index("event_id", drop=False)
    suspicious_nodes: set[str] = set()
    rows: list[dict[str, Any]] = []

    for cid, component in enumerate(nx.connected_components(G)):
        if len(component) < minimum_cluster_size:
            continue
        event_ids = list(component)
        sub = desc.loc[[e for e in event_ids if e in desc.index]]
        if sub.empty:
            continue
        n_vehicles = int(sub["vehicle_model"].nunique())
        mean_score = float(sub["anomaly_score"].mean())
        is_suspicious = (
            len(component) >= minimum_cluster_size
            and n_vehicles >= minimum_vehicle_count
            and mean_score >= fleet_cluster_score_threshold
        )
        rows.append(
            {
                "cluster_id": cid,
                "cluster_method": "connected_component",
                "cluster_size": len(component),
                "num_unique_vehicles": n_vehicles,
                "mean_anomaly_score": round(mean_score, 6),
                "is_fleet_suspicious_cluster": int(is_suspicious),
            }
        )
        if is_suspicious:
            suspicious_nodes.update(event_ids)

    return suspicious_nodes, pd.DataFrame(rows)


def identify_suspicious_clusters(
    G: nx.Graph,
    descriptors: pd.DataFrame,
    *,
    minimum_cluster_size: int,
    minimum_vehicle_count: int,
    similarity_threshold: float,
    fleet_cluster_score_threshold: float,
    cluster_results_path: Path | None = None,
) -> tuple[set[str], pd.DataFrame, str]:
    """
    Return event_ids in fleet-suspicious clusters.

    Uses graph connected components when the similarity graph is not a single
    giant component; otherwise falls back to behavioural DBSCAN/KMeans clusters
    (per spec: connected component *or* cluster).
    """
    n_nodes = max(G.number_of_nodes(), 1)
    components = list(nx.connected_components(G))
    largest = max((len(c) for c in components), default=0)
    degenerate = len(components) == 1 and largest >= _DEGENERATE_GRAPH_FRACTION * n_nodes

    if degenerate and cluster_results_path is not None and cluster_results_path.exists():
        cr = pd.read_csv(cluster_results_path)
        if "algorithm" in cr.columns:
            algo = cr["algorithm"].value_counts().idxmax()
            cr = cr[cr["algorithm"] == algo].copy()
        logger.info(
            "Graph is one connected component (n=%d); using behavioural clusters from %s",
            n_nodes,
            cluster_results_path,
        )
        nodes, summary = identify_suspicious_behavioural_clusters(
            cr,
            minimum_cluster_size=minimum_cluster_size,
            minimum_vehicle_count=minimum_vehicle_count,
            similarity_threshold=similarity_threshold,
            fleet_cluster_score_threshold=fleet_cluster_score_threshold,
        )
        return nodes, summary, "behavioural_clustering"

    nodes, summary = identify_suspicious_connected_components(
        G,
        descriptors,
        minimum_cluster_size=minimum_cluster_size,
        minimum_vehicle_count=minimum_vehicle_count,
        fleet_cluster_score_threshold=fleet_cluster_score_threshold,
    )
    return nodes, summary, "connected_component"


def assign_component_labels(G: nx.Graph) -> dict[str, int]:
    labels: dict[str, int] = {}
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            labels[str(node)] = cid
    return labels


def evaluate_cluster_quality(
    descriptors: pd.DataFrame,
    X: np.ndarray,
    component_labels: dict[str, int],
) -> pd.DataFrame:
    event_ids = descriptors["event_id"].astype(str).tolist()
    cluster_ids = np.array([component_labels.get(eid, -1) for eid in event_ids])
    valid = cluster_ids >= 0
    Xv = X[valid]
    cv = cluster_ids[valid]
    n_clusters = len(np.unique(cv))

    rows: list[dict[str, Any]] = []
    if n_clusters >= 2 and len(cv) > n_clusters:
        try:
            sil = float(silhouette_score(Xv, cv))
        except ValueError:
            sil = float("nan")
        try:
            dbi = float(davies_bouldin_score(Xv, cv))
        except ValueError:
            dbi = float("nan")
        rows.append({"metric": "silhouette_score", "value": sil})
        rows.append({"metric": "davies_bouldin_index", "value": dbi})

    if "ground_truth_label" in descriptors.columns:
        y_true = descriptors.loc[valid, "ground_truth_label"].astype(int).to_numpy()
        try:
            rows.append({"metric": "adjusted_rand_index", "value": float(adjusted_rand_score(y_true, cv))})
            rows.append(
                {
                    "metric": "normalized_mutual_info",
                    "value": float(normalized_mutual_info_score(y_true, cv)),
                }
            )
        except ValueError:
            pass
        # Cluster purity (label homogeneity)
        purities: list[float] = []
        for cid in np.unique(cv):
            mask = cv == cid
            labels = y_true[mask]
            if len(labels):
                purities.append(float(np.max(np.bincount(labels))) / len(labels))
        rows.append(
            {
                "metric": "mean_cluster_purity",
                "value": float(np.mean(purities)) if purities else float("nan"),
            }
        )

    sizes = pd.Series(cv).value_counts()
    rows.append({"metric": "num_clusters", "value": float(n_clusters)})
    rows.append({"metric": "mean_cluster_size", "value": float(sizes.mean()) if len(sizes) else 0.0})
    rows.append({"metric": "max_cluster_size", "value": float(sizes.max()) if len(sizes) else 0.0})
    return pd.DataFrame(rows)


def _plot_fleet_graph(
    G: nx.Graph,
    descriptors: pd.DataFrame,
    suspicious_nodes: set[str],
    out: Path,
    *,
    max_nodes: int,
    seed: int,
) -> bool:
    """Plot subgraph sample; return False if fallback to embedding is recommended."""
    if G.number_of_nodes() > max_nodes * 3:
        return False
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        seed_nodes = set(rng.choice(nodes, size=max_nodes, replace=False))
        sub_nodes = set(seed_nodes)
        for _ in range(max_nodes * 2):
            new = set(sub_nodes)
            for n in list(sub_nodes):
                new.update(G.neighbors(n))
            sub_nodes = new
            if len(sub_nodes) >= max_nodes:
                break
        sub_nodes = set(list(sub_nodes)[:max_nodes])
    else:
        sub_nodes = set(nodes)

    H = G.subgraph(sub_nodes).copy()
    if H.number_of_nodes() < 2:
        return False

    pos = nx.spring_layout(H, seed=seed, k=0.15, iterations=50)
    desc_idx = descriptors.set_index("event_id")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for vehicle, color in VEHICLE_COLORS.items():
        v_nodes = [n for n in H.nodes if n in desc_idx.index and desc_idx.loc[n, "vehicle_model"] == vehicle]
        if not v_nodes:
            continue
        nx.draw_networkx_nodes(
            H,
            pos,
            nodelist=v_nodes,
            node_color=color,
            node_size=18,
            alpha=0.75,
            ax=ax,
            label=vehicle,
        )
    other = [n for n in H.nodes if n in suspicious_nodes]
    if other:
        nx.draw_networkx_nodes(
            H,
            pos,
            nodelist=other,
            node_color="none",
            edgecolors="red",
            linewidths=1.2,
            node_size=28,
            ax=ax,
            label="Fleet-suspicious",
        )
    nx.draw_networkx_edges(H, pos, alpha=0.08, width=0.3, ax=ax)
    ax.set_title(
        f"Fleet Anomaly Graph (n={H.number_of_nodes()}, m={H.number_of_edges()})",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7, markerscale=1.5)
    ax.axis("off")
    _save_figure(fig, out)
    return True


def _plot_descriptor_embedding(
    X: np.ndarray,
    descriptors: pd.DataFrame,
    fleet_pred: np.ndarray,
    out: Path,
    *,
    seed: int,
    max_points: int = 5000,
) -> None:
    n = len(descriptors)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        X = X[idx]
        sub = descriptors.iloc[idx].copy()
        fleet_pred = fleet_pred[idx]
    else:
        sub = descriptors

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    perplexity = min(30, max(5, len(Xs) // 50))
    emb = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    vehicles = sub["vehicle_model"].astype(str).unique()
    for vehicle in vehicles:
        mask = sub["vehicle_model"].astype(str) == vehicle
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.5,
            c=VEHICLE_COLORS.get(vehicle, "#888888"),
            label=vehicle,
        )
    fleet_mask = fleet_pred.astype(bool)
    if fleet_mask.any():
        ax.scatter(
            emb[fleet_mask, 0],
            emb[fleet_mask, 1],
            s=40,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            label="Fleet-suspicious",
        )
    ax.set_title("Descriptor Embedding (t-SNE) — Fleet Correlation")
    ax.legend(loc="best", fontsize=7, markerscale=1.2)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    _save_figure(fig, out)


def _cluster_labels_for_quality(
    descriptors: pd.DataFrame,
    G: nx.Graph,
    cluster_results_path: Path | None,
) -> dict[str, int]:
    """Cluster IDs for quality metrics (behavioural clusters when graph is degenerate)."""
    n_nodes = max(G.number_of_nodes(), 1)
    components = list(nx.connected_components(G))
    largest = max((len(c) for c in components), default=0)
    degenerate = len(components) == 1 and largest >= _DEGENERATE_GRAPH_FRACTION * n_nodes
    if degenerate and cluster_results_path is not None and cluster_results_path.exists():
        cr = pd.read_csv(cluster_results_path)
        if "algorithm" in cr.columns:
            cr = cr[cr["algorithm"] == cr["algorithm"].value_counts().idxmax()]
        return dict(zip(cr["event_id"].astype(str), cr["cluster_id"].astype(int)))
    return assign_component_labels(G)


def run_fleet_correlation_experiment(
    *,
    descriptors_path: Path,
    features_path: Path,
    cluster_results_path: Path | None,
    outputs: FleetCorrelationOutputs,
    cfg: FleetCorrelationConfig,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    descriptors = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    if "ground_truth_label" not in descriptors.columns:
        raise ValueError("Descriptors require ground_truth_label for supervised metrics.")

    y_true = descriptors["ground_truth_label"].astype(int).to_numpy()
    y_score = descriptors["anomaly_score"].astype(float).to_numpy()
    local_pred = descriptors["local_alert"].astype(int).to_numpy()

    local_m = _detection_metrics(y_true, local_pred, y_score)
    pd.DataFrame([{"system": "local_only", **local_m}]).to_csv(
        outputs.results_dir / "local_only_detection_metrics.csv",
        index=False,
    )

    _G_before, G, build_stats, _ = build_fleet_correlation_graph(
        descriptors,
        top_k_neighbors=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        max_nodes=None,
        seed=cfg.graph_viz_seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    logger.info(
        "Built top-k fleet correlation graph: n=%d, edges after=%d, density after=%.6f",
        int(build_stats.get("after_num_nodes", 0)),
        int(build_stats.get("after_num_edges", 0)),
        build_stats.get("after_graph_density", 0.0),
    )

    graph_stats = {
        "number_of_nodes": int(build_stats.get("after_num_nodes", 0)),
        "number_of_edges_before_pruning": int(build_stats.get("before_num_edges", 0)),
        "graph_density_before_pruning": float(build_stats.get("before_graph_density", 0.0)),
        "connected_components_before_pruning": int(
            build_stats.get("before_connected_components", 0)
        ),
        "number_of_edges_after_pruning": int(build_stats.get("after_num_edges", 0)),
        "graph_density_after_pruning": float(build_stats.get("after_graph_density", 0.0)),
        "number_of_edges": int(build_stats.get("after_num_edges", 0)),
        "average_degree": float(build_stats.get("after_average_degree", 0.0)),
        "graph_density": float(build_stats.get("after_graph_density", 0.0)),
        "connected_components": int(build_stats.get("after_connected_components", 0)),
        "largest_component_size": int(
            max((len(c) for c in nx.connected_components(G)), default=0)
        ),
        "top_k_neighbors": cfg.top_k_neighbors,
        "similarity_threshold": cfg.similarity_threshold,
        "minimum_vehicle_count": cfg.minimum_vehicle_count,
        "fleet_cluster_score_threshold": cfg.fleet_cluster_score_threshold,
    }
    pd.DataFrame([graph_stats]).to_csv(
        outputs.results_dir / "fleet_graph_statistics.csv",
        index=False,
    )

    suspicious_nodes, cluster_df, cluster_method = identify_suspicious_clusters(
        G,
        descriptors,
        minimum_cluster_size=cfg.minimum_cluster_size,
        minimum_vehicle_count=cfg.minimum_vehicle_count,
        similarity_threshold=cfg.similarity_threshold,
        fleet_cluster_score_threshold=cfg.fleet_cluster_score_threshold,
        cluster_results_path=cluster_results_path,
    )
    cluster_df.to_csv(outputs.results_dir / "fleet_suspicious_clusters.csv", index=False)

    in_cluster = descriptors["event_id"].astype(str).isin(suspicious_nodes).to_numpy()
    fleet_pred = ((local_pred == 1) | in_cluster).astype(int)
    fleet_m = _detection_metrics(y_true, fleet_pred, y_score)

    pred_out = descriptors[
        [
            "event_id",
            "window_id",
            "vehicle_model",
            "attack_type",
            "anomaly_score",
            "local_alert",
            "ground_truth_label",
        ]
    ].copy()
    pred_out["in_fleet_suspicious_cluster"] = in_cluster.astype(int)
    pred_out["fleet_prediction"] = fleet_pred
    pred_out.to_csv(outputs.results_dir / "fleet_level_predictions.csv", index=False)

    pd.DataFrame([{"system": "fleet_aware", **fleet_m}]).to_csv(
        outputs.results_dir / "fleet_level_detection_metrics.csv",
        index=False,
    )

    improvement = pd.DataFrame(
        [
            {
                "metric": "detection_gain_percent",
                "local_value": local_m["detection_score"],
                "fleet_value": fleet_m["detection_score"],
                "gain_percent": _pct_gain(fleet_m["detection_score"], local_m["detection_score"]),
            },
            {
                "metric": "recall_gain_percent",
                "local_value": local_m["recall"],
                "fleet_value": fleet_m["recall"],
                "gain_percent": _pct_gain(fleet_m["recall"], local_m["recall"]),
            },
            {
                "metric": "f1_gain_percent",
                "local_value": local_m["f1"],
                "fleet_value": fleet_m["f1"],
                "gain_percent": _pct_gain(fleet_m["f1"], local_m["f1"]),
            },
            {
                "metric": "roc_auc_gain_percent",
                "local_value": local_m["roc_auc"],
                "fleet_value": fleet_m["roc_auc"],
                "gain_percent": _pct_gain(fleet_m["roc_auc"], local_m["roc_auc"]),
            },
        ]
    )
    improvement.to_csv(outputs.results_dir / "fleet_correlation_improvement.csv", index=False)

    X, _ = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    component_labels = _cluster_labels_for_quality(descriptors, G, cluster_results_path)
    cluster_quality = evaluate_cluster_quality(descriptors, X, component_labels)
    cluster_quality.to_csv(outputs.results_dir / "fleet_cluster_quality.csv", index=False)

    comparison_rows = []
    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("false_positive_rate", "False Positive Rate"),
        ("detection_score", "Detection Score"),
    ]:
        lv, fv = local_m[key], fleet_m[key]
        comparison_rows.append(
            {
                "Metric": label,
                "Local IDS": round(lv, 4),
                "Fleet-Aware IDS": round(fv, 4),
                "Improvement (%)": round(_pct_gain(fv, lv), 2),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    (outputs.tables_dir / "table_local_vs_fleet_ids.tex").write_text(
        _df_to_ieee_tex(
            comparison,
            "Local IDS vs fleet-aware IDS on anomaly descriptors.",
            "tab:local-vs-fleet",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_local_vs_fleet_ids.md").write_text(
        "# Local IDS vs Fleet-Aware IDS\n\n" + _df_to_markdown(comparison),
        encoding="utf-8",
    )

    graph_fig = outputs.figures_dir / "fleet_anomaly_graph"
    embed_fig = outputs.figures_dir / "descriptor_embedding_fleet_correlation"
    graph_ok = _plot_fleet_graph(
        G,
        descriptors,
        suspicious_nodes,
        graph_fig,
        max_nodes=cfg.max_graph_viz_nodes,
        seed=cfg.graph_viz_seed,
    )
    primary_figure = graph_fig
    if not graph_ok:
        _plot_descriptor_embedding(
            X,
            descriptors,
            fleet_pred,
            embed_fig,
            seed=cfg.graph_viz_seed,
        )
        primary_figure = embed_fig

    n_promoted = int(((fleet_pred == 1) & (local_pred == 0)).sum())
    f1_gain = _pct_gain(fleet_m["f1"], local_m["f1"])
    fleet_improves = fleet_m["detection_score"] > local_m["detection_score"]
    fpr_delta = fleet_m["false_positive_rate"] - local_m["false_positive_rate"]
    summary_path = outputs.results_dir / "fleet_correlation_summary.md"
    if fleet_improves and fpr_delta <= 0.01:
        conclusion = (
            "**Fleet-level graph reasoning improves detection beyond single-vehicle IDS** "
            f"(F1 gain {f1_gain:.2f}%, recall gain {_pct_gain(fleet_m['recall'], local_m['recall']):.2f}%) "
            "with negligible FPR increase."
        )
    elif fleet_improves:
        conclusion = (
            "**Fleet-level graph reasoning improves F1 beyond single-vehicle IDS**, "
            f"but fleet FPR increased by {fpr_delta:.4f} — report both gains and the FPR trade-off in the paper."
        )
    else:
        conclusion = (
            "**Fleet-level graph reasoning did not improve F1 over local-only IDS** "
            "under the current thresholds; weak-signal promotions are listed in fleet_level_predictions.csv."
        )
    summary_path.write_text(
        "\n".join(
            [
                "# Fleet Correlation Experiment — Summary",
                "",
                "## Detection performance",
                f"- **Local-only detection score (F1):** {local_m['detection_score']:.4f}",
                f"- **Fleet-aware detection score (F1):** {fleet_m['detection_score']:.4f}",
                f"- **Detection gain (F1):** {_pct_gain(fleet_m['detection_score'], local_m['detection_score']):.2f}%",
                f"- **Recall gain:** {_pct_gain(fleet_m['recall'], local_m['recall']):.2f}%",
                f"- **F1 gain:** {f1_gain:.2f}%",
                f"- **ROC-AUC gain:** {_pct_gain(fleet_m['roc_auc'], local_m['roc_auc']):.2f}%",
                f"- **Windows promoted by fleet correlation (not local_alert):** {n_promoted}",
                "",
                "## Fleet graph",
                f"- **Nodes:** {graph_stats['number_of_nodes']}",
                f"- **Edges:** {graph_stats['number_of_edges']}",
                f"- **Average degree:** {graph_stats['average_degree']:.2f}",
                f"- **Connected components:** {graph_stats['connected_components']}",
                f"- **Largest component:** {graph_stats['largest_component_size']}",
                f"- **Edges before pruning (top-k union):** {graph_stats['number_of_edges_before_pruning']}",
                f"- **Density before pruning:** {graph_stats['graph_density_before_pruning']:.8f}",
                f"- **Edges after pruning (sim ≥ {cfg.similarity_threshold}):** {graph_stats['number_of_edges_after_pruning']}",
                f"- **Density after pruning:** {graph_stats['graph_density_after_pruning']:.8f}",
                f"- **Components before pruning:** {graph_stats['connected_components_before_pruning']}",
                f"- **Fleet cluster method:** {cluster_method}",
                "",
                "## Cluster quality",
                *[f"- **{row['metric']}:** {row['value']:.4f}" for _, row in cluster_quality.iterrows()],
                "",
                "## False positives",
                f"- **Local FPR:** {local_m['false_positive_rate']:.4f}",
                f"- **Fleet FPR:** {fleet_m['false_positive_rate']:.4f}",
                f"- **FPR change (fleet − local):** {fpr_delta:+.4f}",
                "",
                "## Interpretation",
                "",
                conclusion,
                "",
                "Fleet graph uses top-k cosine neighbours per descriptor (k="
                f"{cfg.top_k_neighbors}) with similarity ≥ {cfg.similarity_threshold}; "
                "suspicious groups are connected components with ≥ "
                f"{cfg.minimum_vehicle_count} vehicles and mean anomaly score ≥ "
                f"{cfg.fleet_cluster_score_threshold} (behavioural clustering fallback if degenerate).",
                "",
                "Parameters: "
                f"top_k_neighbors={cfg.top_k_neighbors}, "
                f"similarity_threshold={cfg.similarity_threshold}, "
                f"minimum_cluster_size={cfg.minimum_cluster_size}, "
                f"minimum_vehicle_count={cfg.minimum_vehicle_count}, "
                f"fleet_cluster_score_threshold={cfg.fleet_cluster_score_threshold}.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    logger.info("Fleet correlation experiment complete.")
    return {
        "fleet_correlation_summary": summary_path,
        "table_local_vs_fleet": outputs.tables_dir / "table_local_vs_fleet_ids.tex",
        "primary_figure": primary_figure.with_suffix(".png"),
    }
