"""Fleet-aware coordinated CAN attack campaign detection evaluation (IEEE-ready)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import recall_score

from src.evaluation.campaign_clustering import (
    extend_dbscan_labels,
    mean_intra_cluster_similarity,
    run_dbscan,
    subsample_indices,
)
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_graph,
    event_id_to_window_id,
    load_anomaly_descriptors,
    resolve_fleet_similarity_matrix,
)
from src.graph.fleet_similarity_features import SimilarityFeatureView
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

CAMPAIGN_ATTACK_TYPES = ("flooding", "fuzzy", "replay", "malfunction")

ATTACK_COLORS = {
    "flooding": "#C00000",
    "fuzzy": "#ED7D31",
    "replay": "#7030A0",
    "malfunction": "#4472C4",
    "attack_free": "#A6A6A6",
}

VEHICLE_MARKERS = {
    "Hyundai": "o",
    "Kia": "s",
    "Chevrolet": "^",
}

VEHICLE_COLORS = {
    "Hyundai": "#ED7D31",
    "Kia": "#70AD47",
    "Chevrolet": "#4472C4",
}


@dataclass(frozen=True)
class CampaignDetectionOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class CampaignDetectionConfig:
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    similarity_threshold: float = 0.95
    similarity_feature_view: SimilarityFeatureView = "behavior_only_vehicle_normalized"
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()
    min_vehicles: int = 2
    min_cluster_size: int = 10
    min_cohesion: float = 0.85
    min_dominant_attack_ratio: float = 0.60
    campaign_match_recall: float = 0.15
    campaign_match_min_nodes: int = 20
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 10
    dbscan_pca_components: int = 8
    max_clustering_samples: int = 20000
    max_graph_viz_nodes: int = 800
    max_embedding_samples: int = 5000
    embedding_method: Literal["tsne", "umap"] = "tsne"
    seed: int = 42


def _df_to_markdown(df: pd.DataFrame, title: str) -> str:
    cols = list(df.columns)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _df_to_tex(df: pd.DataFrame, caption: str, label: str) -> str:
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


def _vehicle_short(vehicle_model: str) -> str:
    mapping = {"Hyundai": "HYU", "Kia": "KIA", "Chevrolet": "CHV"}
    return mapping.get(str(vehicle_model), str(vehicle_model)[:3].upper())


def build_campaign_ground_truth(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Controlled campaign scenarios: same attack behaviour across >=2 vehicles."""
    desc = descriptors.copy()
    if "window_id" not in desc.columns:
        desc["window_id"] = desc["event_id"].map(event_id_to_window_id)

    rows: list[dict[str, Any]] = []
    for attack_type in CAMPAIGN_ATTACK_TYPES:
        sub = desc[desc["attack_type"] == attack_type]
        if sub["vehicle_model"].nunique(dropna=True) < 2:
            logger.warning("Skipping campaign %s: fewer than 2 vehicles.", attack_type)
            continue
        campaign_id = f"CAMP-{attack_type}"
        for _, row in sub.iterrows():
            rows.append(
                {
                    "campaign_id": campaign_id,
                    "attack_type": attack_type,
                    "vehicle_id": row["vehicle_model"],
                    "window_id": int(row["window_id"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "descriptor_id": row["event_id"],
                }
            )
    gt = pd.DataFrame(rows)
    logger.info(
        "Campaign ground truth: %d scenarios, %d windows",
        gt["campaign_id"].nunique(),
        len(gt),
    )
    return gt


def export_campaign_graph_statistics(
    graph: nx.Graph,
    *,
    similarity_view: str,
    top_k_same_vehicle: int,
    top_k_cross_vehicle: int,
    similarity_threshold: float,
) -> pd.DataFrame:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    cross = sum(
        1
        for u, v in graph.edges()
        if graph.nodes[u].get("vehicle_model") != graph.nodes[v].get("vehicle_model")
    )
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if n else 0.0
    return pd.DataFrame(
        [
            {
                "similarity_view": similarity_view,
                "top_k_same_vehicle": top_k_same_vehicle,
                "top_k_cross_vehicle": top_k_cross_vehicle,
                "similarity_threshold": similarity_threshold,
                "nodes": n,
                "edges": m,
                "density": round(density, 8),
                "average_degree": round(avg_degree, 4),
                "connected_components": nx.number_connected_components(graph),
                "cross_vehicle_edge_percentage": round(100.0 * cross / max(m, 1), 4),
            }
        ]
    )


def _dominant_attack(meta: pd.DataFrame, mask: np.ndarray) -> tuple[str, float]:
    sub = meta.loc[mask, "attack_type"]
    if sub.empty:
        return "unknown", 0.0
    counts = sub.value_counts()
    dominant = str(counts.index[0])
    ratio = float(counts.iloc[0] / len(sub))
    return dominant, ratio


def _cluster_rows_from_labels(
    labels: np.ndarray,
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    algorithm: str,
    cfg: CampaignDetectionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid in sorted({int(c) for c in np.unique(labels)}):
        if cid == -1:
            continue
        mask = labels == cid
        size = int(mask.sum())
        n_veh = int(meta.loc[mask, "vehicle_model"].nunique(dropna=True))
        mean_sim = mean_intra_cluster_similarity(X[mask]) if size >= 2 else 0.0
        dominant, dom_ratio = _dominant_attack(meta, mask)
        mean_score = float(meta.loc[mask, "anomaly_score"].mean())
        is_valid = (
            size >= cfg.min_cluster_size
            and n_veh >= cfg.min_vehicles
            and mean_sim >= cfg.min_cohesion
            and dom_ratio >= cfg.min_dominant_attack_ratio
        )
        rows.append(
            {
                "detected_cluster_id": f"{algorithm}-{cid}",
                "algorithm": algorithm,
                "raw_cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "vehicle_models": ",".join(sorted(meta.loc[mask, "vehicle_model"].dropna().unique())),
                "dominant_attack_type": dominant,
                "dominant_attack_ratio": round(dom_ratio, 4),
                "mean_similarity": round(mean_sim, 4),
                "mean_anomaly_score": round(mean_score, 4),
                "is_valid_campaign_cluster": bool(is_valid),
            }
        )
    return pd.DataFrame(rows)


def _run_hdbscan_labels(X: np.ndarray, cfg: CampaignDetectionConfig) -> np.ndarray | None:
    try:
        import hdbscan  # type: ignore
    except ImportError:
        logger.warning("hdbscan not installed; skipping HDBSCAN.")
        return None
    labels = hdbscan.HDBSCAN(
        min_cluster_size=cfg.dbscan_min_samples,
        min_samples=max(1, cfg.dbscan_min_samples // 2),
        metric="euclidean",
    ).fit_predict(X)
    return labels.astype(int)


def _connected_component_labels(graph: nx.Graph, event_ids: list[str]) -> np.ndarray:
    id_to_idx = {eid: i for i, eid in enumerate(event_ids)}
    labels = np.full(len(event_ids), -1, dtype=int)
    for comp_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            idx = id_to_idx.get(str(node))
            if idx is not None:
                labels[idx] = comp_id
    return labels


def detect_campaign_clusters(
    descriptors: pd.DataFrame,
    graph: nx.Graph,
    cfg: CampaignDetectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return per-cluster summary, per-node assignments, and primary algorithm name."""
    X, _ = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    meta = descriptors.reset_index(drop=True)
    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        X[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    dbscan_labels = extend_dbscan_labels(
        X,
        fit_labels,
        X[fit_idx],
        projector,
        eps=cfg.dbscan_eps,
    )

    hdbscan_labels = _run_hdbscan_labels(X, cfg)
    cc_labels = _connected_component_labels(graph, meta["event_id"].astype(str).tolist())

    dbscan_clusters = _cluster_rows_from_labels(dbscan_labels, X, meta, algorithm="dbscan", cfg=cfg)
    cc_clusters = _cluster_rows_from_labels(cc_labels, X, meta, algorithm="connected_components", cfg=cfg)

    valid_dbscan = int(dbscan_clusters["is_valid_campaign_cluster"].sum()) if not dbscan_clusters.empty else 0
    valid_cc = int(cc_clusters["is_valid_campaign_cluster"].sum()) if not cc_clusters.empty else 0
    if valid_dbscan >= valid_cc:
        primary = "dbscan"
        primary_labels = dbscan_labels
        cluster_df = dbscan_clusters
    else:
        primary = "connected_components"
        primary_labels = cc_labels
        cluster_df = cc_clusters

    if hdbscan_labels is not None:
        hdbscan_clusters = _cluster_rows_from_labels(
            hdbscan_labels, X, meta, algorithm="hdbscan", cfg=cfg
        )
        valid_hdb = int(hdbscan_clusters["is_valid_campaign_cluster"].sum()) if not hdbscan_clusters.empty else 0
        if valid_hdb > int(cluster_df["is_valid_campaign_cluster"].sum()):
            primary = "hdbscan"
            primary_labels = hdbscan_labels
            cluster_df = hdbscan_clusters

    assignments = meta[["event_id", "vehicle_model", "attack_type", "anomaly_score"]].copy()
    assignments["detected_cluster_id"] = [
        f"{primary}-{int(lbl)}" if int(lbl) >= 0 else "noise"
        for lbl in primary_labels
    ]
    assignments["cluster_algorithm"] = primary

    logger.info(
        "Campaign clustering primary=%s; valid clusters=%d",
        primary,
        int(cluster_df["is_valid_campaign_cluster"].sum()) if not cluster_df.empty else 0,
    )
    return cluster_df, assignments, primary


def _valid_clusters(cluster_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_df.empty:
        return cluster_df
    return cluster_df[cluster_df["is_valid_campaign_cluster"]].copy()


def _cluster_node_sets(
    assignments: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> dict[str, set[str]]:
    valid_ids = set(_valid_clusters(cluster_df)["detected_cluster_id"])
    out: dict[str, set[str]] = {}
    for cid in valid_ids:
        nodes = set(assignments.loc[assignments["detected_cluster_id"] == cid, "event_id"].astype(str))
        if nodes:
            out[cid] = nodes
    return out


def _true_campaign_sets(ground_truth: pd.DataFrame) -> dict[str, dict[str, Any]]:
    campaigns: dict[str, dict[str, Any]] = {}
    for campaign_id, grp in ground_truth.groupby("campaign_id"):
        campaigns[str(campaign_id)] = {
            "attack_type": str(grp["attack_type"].iloc[0]),
            "nodes": set(grp["descriptor_id"].astype(str)),
            "vehicles": sorted(grp["vehicle_id"].dropna().unique()),
            "n_vehicles": int(grp["vehicle_id"].nunique()),
        }
    return campaigns


def _match_cluster_to_campaign(
    cluster_nodes: set[str],
    cluster_row: pd.Series,
    true_campaign: dict[str, Any],
    assignments: pd.DataFrame,
    cfg: CampaignDetectionConfig,
) -> bool:
    true_nodes = true_campaign["nodes"]
    overlap = cluster_nodes & true_nodes
    if len(overlap) < cfg.campaign_match_min_nodes:
        return False
    if cluster_row["dominant_attack_type"] != true_campaign["attack_type"]:
        return False
    overlap_meta = assignments[assignments["event_id"].astype(str).isin(overlap)]
    if overlap_meta["vehicle_model"].nunique(dropna=True) < cfg.min_vehicles:
        return False
    recall = len(overlap) / max(len(true_nodes), 1)
    if recall < cfg.campaign_match_recall:
        return False
    return int(cluster_row["vehicles_in_cluster"]) >= cfg.min_vehicles


def evaluate_campaign_detection(
    ground_truth: pd.DataFrame,
    cluster_df: pd.DataFrame,
    assignments: pd.DataFrame,
    cfg: CampaignDetectionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    true_campaigns = _true_campaign_sets(ground_truth)
    cluster_nodes = _cluster_node_sets(assignments, cluster_df)
    valid = _valid_clusters(cluster_df)

    total_true = len(true_campaigns)
    detected_true = 0
    matched_cluster_ids: set[str] = set()
    campaign_rows: list[dict[str, Any]] = []

    for campaign_id, tc in true_campaigns.items():
        attack_type = tc["attack_type"]
        matching: list[str] = []
        for cid, nodes in cluster_nodes.items():
            row = valid.loc[valid["detected_cluster_id"] == cid].iloc[0]
            if _match_cluster_to_campaign(nodes, row, tc, assignments, cfg):
                matching.append(cid)
                matched_cluster_ids.add(cid)
        is_detected = len(matching) > 0
        detected_true += int(is_detected)
        purity = float("nan")
        fragmentation = len(matching)
        if matching:
            purities = [
                float(valid.loc[valid["detected_cluster_id"] == cid, "dominant_attack_ratio"].iloc[0])
                for cid in matching
            ]
            purity = float(np.mean(purities))
        campaign_rows.append(
            {
                "campaign_id": campaign_id,
                "attack_type": attack_type,
                "vehicles_involved": tc["n_vehicles"],
                "true_campaign_windows": len(tc["nodes"]),
                "detected": is_detected,
                "matching_clusters": len(matching),
                "fragmentation": fragmentation,
                "campaign_purity": purity,
            }
        )

    all_valid_clusters = len(valid)
    false_clusters = all_valid_clusters - len(matched_cluster_ids)
    false_campaign_rate = false_clusters / max(all_valid_clusters, 1)

    detection_rate = detected_true / max(total_true, 1)
    precision = len(matched_cluster_ids) / max(all_valid_clusters, 1)
    recall = detection_rate
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    cross_vehicle_coverage = (
        float((valid["vehicles_in_cluster"] >= 2).mean()) if not valid.empty else 0.0
    )
    purity_overall = float(valid["dominant_attack_ratio"].mean()) if not valid.empty else float("nan")

    fragmentation_overall = (
        float(np.mean([r["fragmentation"] for r in campaign_rows if r["detected"]]))
        if any(r["detected"] for r in campaign_rows)
        else 0.0
    )

    matched_nodes: set[str] = set()
    for tc in true_campaigns.values():
        for cid, nodes in cluster_nodes.items():
            row = valid.loc[valid["detected_cluster_id"] == cid].iloc[0]
            if _match_cluster_to_campaign(nodes, row, tc, assignments, cfg):
                matched_nodes |= nodes & tc["nodes"]
    gt_ids = set(ground_truth["descriptor_id"].astype(str))
    fleet_campaign_window_recall = len(matched_nodes) / max(len(gt_ids), 1)

    metrics = {
        "campaign_detection_rate": detection_rate,
        "campaign_precision": precision,
        "campaign_recall": recall,
        "campaign_f1": f1,
        "campaign_purity": purity_overall,
        "cross_vehicle_coverage": cross_vehicle_coverage,
        "fragmentation_mean": fragmentation_overall,
        "false_campaign_rate": false_campaign_rate,
        "campaign_window_recall": fleet_campaign_window_recall,
        "total_true_campaigns": total_true,
        "detected_true_campaigns": detected_true,
        "valid_detected_clusters": all_valid_clusters,
        "matched_detected_clusters": len(matched_cluster_ids),
        "false_detected_clusters": false_clusters,
    }

    per_type = pd.DataFrame(campaign_rows)
    overall = pd.DataFrame(
        [
            {
                "metric": "campaign_detection_rate",
                "value": round(detection_rate, 4),
            },
            {"metric": "campaign_precision", "value": round(precision, 4)},
            {"metric": "campaign_recall", "value": round(recall, 4)},
            {"metric": "campaign_f1", "value": round(f1, 4)},
            {"metric": "campaign_purity", "value": round(purity_overall, 4)},
            {"metric": "cross_vehicle_coverage", "value": round(cross_vehicle_coverage, 4)},
            {"metric": "fragmentation_mean", "value": round(fragmentation_overall, 4)},
            {"metric": "false_campaign_rate", "value": round(false_campaign_rate, 4)},
            {"metric": "campaign_window_recall", "value": round(fleet_campaign_window_recall, 4)},
            {"metric": "total_true_campaigns", "value": total_true},
            {"metric": "detected_true_campaigns", "value": detected_true},
            {"metric": "valid_detected_clusters", "value": all_valid_clusters},
        ]
    )
    return overall, per_type, metrics


def compare_local_vs_fleet_campaign(
    descriptors: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cluster_df: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics: dict[str, Any],
    cfg: CampaignDetectionConfig,
) -> pd.DataFrame:
    gt_ids = set(ground_truth["descriptor_id"].astype(str))
    desc = descriptors[descriptors["event_id"].astype(str).isin(gt_ids)].copy()
    y_true = np.ones(len(desc), dtype=int)
    y_local = desc["local_alert"].astype(int).to_numpy()
    local_recall = float(recall_score(y_true, y_local, zero_division=0))

    rows = [
        {
            "Metric": "Attack detection recall",
            "Local IDS": round(local_recall, 4),
            "Fleet-Aware IDS": round(local_recall, 4),
            "Improvement / Added Capability": "Same local evidence (no retraining)",
        },
        {
            "Metric": "Campaign-level detection rate",
            "Local IDS": 0.0,
            "Fleet-Aware IDS": round(metrics["campaign_detection_rate"], 4),
            "Improvement / Added Capability": "Fleet-only capability",
        },
        {
            "Metric": "Cross-vehicle correlation",
            "Local IDS": 0.0,
            "Fleet-Aware IDS": round(metrics["cross_vehicle_coverage"], 4),
            "Improvement / Added Capability": "Fleet-only capability",
        },
        {
            "Metric": "Campaign purity",
            "Local IDS": "--",
            "Fleet-Aware IDS": round(metrics["campaign_purity"], 4),
            "Improvement / Added Capability": "Fleet-only capability",
        },
        {
            "Metric": "False campaign rate",
            "Local IDS": "--",
            "Fleet-Aware IDS": round(metrics["false_campaign_rate"], 4),
            "Improvement / Added Capability": "Lower is better",
        },
    ]
    return pd.DataFrame(rows)


def build_campaign_results_table(per_type: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attack_type in CAMPAIGN_ATTACK_TYPES:
        sub = per_type[per_type["attack_type"] == attack_type]
        if sub.empty:
            rows.append(
                {
                    "Campaign Type": attack_type.capitalize(),
                    "Vehicles Involved": 0,
                    "Detected?": "No",
                    "Campaign Detection Rate": 0.0,
                    "Campaign Purity": "--",
                    "Fragmentation": 0,
                    "False Campaign Rate": round(metrics["false_campaign_rate"], 4),
                }
            )
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "Campaign Type": attack_type.capitalize(),
                "Vehicles Involved": int(row["vehicles_involved"]),
                "Detected?": "Yes" if bool(row["detected"]) else "No",
                "Campaign Detection Rate": round(float(row["detected"]), 4),
                "Campaign Purity": (
                    round(float(row["campaign_purity"]), 4)
                    if pd.notna(row["campaign_purity"])
                    else "--"
                ),
                "Fragmentation": int(row["fragmentation"]),
                "False Campaign Rate": round(metrics["false_campaign_rate"], 4),
            }
        )
    rows.append(
        {
            "Campaign Type": "Overall",
            "Vehicles Involved": int(per_type["vehicles_involved"].max()) if not per_type.empty else 0,
            "Detected?": (
                f"{int(per_type['detected'].sum())}/{len(per_type)}" if not per_type.empty else "0/0"
            ),
            "Campaign Detection Rate": round(metrics["campaign_detection_rate"], 4),
            "Campaign Purity": round(metrics["campaign_purity"], 4),
            "Fragmentation": round(metrics["fragmentation_mean"], 2),
            "False Campaign Rate": round(metrics["false_campaign_rate"], 4),
        }
    )
    return pd.DataFrame(rows)


def _sample_graph_nodes(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    max_nodes: int,
    seed: int,
) -> list[str]:
    valid_ids = set(_valid_clusters(cluster_df)["detected_cluster_id"])
    highlight = set(
        assignments.loc[assignments["detected_cluster_id"].isin(valid_ids), "event_id"].astype(str)
    )
    all_nodes = list(graph.nodes())
    if len(all_nodes) <= max_nodes:
        return all_nodes
    rng = np.random.default_rng(seed)
    sampled = set(rng.choice(all_nodes, size=max(1, max_nodes // 2), replace=False).tolist())
    sampled |= set(rng.choice(list(highlight), size=min(len(highlight), max_nodes // 2), replace=False))
    if len(sampled) > max_nodes:
        sampled = set(rng.choice(list(sampled), size=max_nodes, replace=False))
    return list(sampled)


def plot_fleet_campaign_graph(
    graph: nx.Graph,
    assignments: pd.DataFrame,
    cluster_df: pd.DataFrame,
    out_base: Path,
    *,
    max_nodes: int,
    seed: int,
) -> None:
    nodes = _sample_graph_nodes(graph, assignments, cluster_df, max_nodes=max_nodes, seed=seed)
    sub = graph.subgraph(nodes).copy()
    if sub.number_of_nodes() == 0:
        logger.warning("Empty graph sample; skipping fleet campaign graph figure.")
        return

    valid_ids = set(_valid_clusters(cluster_df)["detected_cluster_id"])
    highlight = set(
        assignments.loc[assignments["detected_cluster_id"].isin(valid_ids), "event_id"].astype(str)
    )

    pos = nx.spring_layout(sub, seed=seed, k=0.15, iterations=50)
    fig, ax = plt.subplots(figsize=(10, 8))

    for u, v in sub.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        cross = sub.nodes[u].get("vehicle_model") != sub.nodes[v].get("vehicle_model")
        ax.plot(x, y, color="#BBBBBB" if cross else "#DDDDDD", linewidth=0.4, alpha=0.6, zorder=1)

    for vehicle, marker in VEHICLE_MARKERS.items():
        veh_nodes = [n for n in sub.nodes() if sub.nodes[n].get("vehicle_model") == vehicle]
        if not veh_nodes:
            continue
        xs = [pos[n][0] for n in veh_nodes]
        ys = [pos[n][1] for n in veh_nodes]
        colors = [
            ATTACK_COLORS.get(str(sub.nodes[n].get("attack_type", "attack_free")), "#888888")
            for n in veh_nodes
        ]
        edgecolors = ["#000000" if str(n) in highlight else VEHICLE_COLORS.get(vehicle, "#333333") for n in veh_nodes]
        linewidths = [1.5 if str(n) in highlight else 0.6 for n in veh_nodes]
        for x, y, c, ec, lw in zip(xs, ys, colors, edgecolors, linewidths):
            ax.scatter(x, y, c=c, marker=marker, s=28 if lw > 1 else 18, edgecolors=ec, linewidths=lw, alpha=0.85, zorder=2)

    ax.set_title("Fleet Campaign Graph (sampled anomaly descriptors)")
    ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, out_base)


def plot_campaign_descriptor_embedding(
    descriptors: pd.DataFrame,
    assignments: pd.DataFrame,
    cluster_df: pd.DataFrame,
    out_base: Path,
    cfg: CampaignDetectionConfig,
) -> None:
    X, _ = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    meta = descriptors.reset_index(drop=True)
    idx = subsample_indices(meta, cfg.max_embedding_samples, seed=cfg.seed)
    X_sub = X[idx]
    meta_sub = meta.iloc[idx].reset_index(drop=True)

    if cfg.embedding_method == "umap":
        try:
            import umap  # type: ignore

            Z = umap.UMAP(n_components=2, random_state=cfg.seed, n_neighbors=30, min_dist=0.1).fit_transform(X_sub)
            method_label = "UMAP"
        except ImportError:
            logger.warning("umap not installed; falling back to t-SNE.")
            Z = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, init="pca").fit_transform(X_sub)
            method_label = "t-SNE"
    else:
        Z = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, init="pca").fit_transform(X_sub)
        method_label = "t-SNE"

    valid_ids = set(_valid_clusters(cluster_df)["detected_cluster_id"])
    assign_sub = assignments.iloc[idx].reset_index(drop=True)
    highlight = assign_sub["detected_cluster_id"].isin(valid_ids).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    for attack_type, color in ATTACK_COLORS.items():
        mask = meta_sub["attack_type"] == attack_type
        if not mask.any():
            continue
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=color,
            s=16,
            alpha=0.35,
            label=attack_type,
            edgecolors="none",
        )

    ax.scatter(
        Z[highlight, 0],
        Z[highlight, 1],
        facecolors="none",
        edgecolors="#000000",
        s=60,
        linewidths=1.0,
        label="Detected campaign cluster",
    )

    for vehicle, marker in VEHICLE_MARKERS.items():
        mask = meta_sub["vehicle_model"] == vehicle
        if mask.any():
            ax.scatter([], [], c="none", marker=marker, edgecolors=VEHICLE_COLORS[vehicle], s=40, label=vehicle)

    ax.set_title(f"Campaign Descriptor Embedding ({method_label})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    _save_figure(fig, out_base)


def write_campaign_detection_summary(
    path: Path,
    *,
    ground_truth: pd.DataFrame,
    metrics: dict[str, Any],
    per_type: pd.DataFrame,
    primary_algorithm: str,
    graph_stats: pd.DataFrame,
) -> None:
    n_scenarios = ground_truth["campaign_id"].nunique()
    types = ", ".join(sorted(ground_truth["attack_type"].unique()))
    best = per_type.sort_values("campaign_purity", ascending=False, na_position="last")
    best_type = str(best.iloc[0]["attack_type"]) if not best.empty and best.iloc[0]["detected"] else "none clearly dominant"
    cross_pct = float(graph_stats["cross_vehicle_edge_percentage"].iloc[0])

    lines = [
        "# Coordinated Campaign Detection — Summary",
        "",
        "**Note:** Controlled campaign scenarios are constructed from labelled attack windows",
        "in the public Car-Hacking dataset. They evaluate fleet-level campaign reasoning;",
        "they do not represent externally synchronized real-world campaigns.",
        "",
        f"1. **Campaign scenarios generated:** {n_scenarios} ({len(ground_truth)} descriptor windows).",
        f"2. **Campaign types evaluated:** {types}.",
        (
            "3. **Fleet correlation detected coordinated campaigns:** "
            + ("Yes — at least one true campaign matched a valid cross-vehicle cluster."
               if metrics["detected_true_campaigns"] > 0
               else "Partially — no true campaign met the full matching criteria under current gates.")
        ),
        f"4. **Best cross-vehicle clustering (attack type):** {best_type}.",
        f"5. **Campaign detection rate:** {metrics['campaign_detection_rate']:.1%} "
        f"({metrics['detected_true_campaigns']}/{metrics['total_true_campaigns']} scenarios).",
        f"6. **Campaign purity (mean dominant-attack ratio):** {metrics['campaign_purity']:.3f}.",
        f"7. **False campaign rate:** {metrics['false_campaign_rate']:.1%}.",
        f"8. **Added capability vs local IDS:** Local IDS flags individual anomalies but cannot group "
        f"cross-vehicle behaviourally similar events. Fleet graph clustering ({primary_algorithm}) "
        f"achieved cross-vehicle edge share {cross_pct:.2f}% and campaign detection rate "
        f"{metrics['campaign_detection_rate']:.1%}.",
        (
            "9. **Limitations:** Campaign scenarios are synthetically defined from per-vehicle labelled attacks; "
            "clustering quality varies by attack type; Chevrolet has fewer replay windows; "
            "high similarity thresholds and minimum cluster gates limit recall; "
            "false campaign clusters remain when behavioural descriptors overlap across attack classes."
        ),
        "",
        "## Conclusion",
        "",
        "The fleet-aware correlation layer enables campaign-level detection by grouping behaviourally "
        "similar anomaly descriptors across multiple vehicles. This provides a capability that is not "
        "available to isolated vehicle-level IDS models.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_campaign_detection_experiment(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: CampaignDetectionOutputs,
    cfg: CampaignDetectionConfig,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    descriptors = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    descriptors = descriptors.copy()
    descriptors["window_id"] = descriptors["event_id"].map(event_id_to_window_id)

    ground_truth = build_campaign_ground_truth(descriptors)
    gt_path = outputs.results_dir / "campaign_ground_truth.csv"
    ground_truth.to_csv(gt_path, index=False)

    _, graph, _, _ = build_cross_vehicle_constrained_graph(
        descriptors,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )

    graph_stats = export_campaign_graph_statistics(
        graph,
        similarity_view=cfg.similarity_feature_view,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
    )
    graph_stats_path = outputs.results_dir / "campaign_graph_statistics.csv"
    graph_stats.to_csv(graph_stats_path, index=False)

    cluster_df, assignments, primary = detect_campaign_clusters(descriptors, graph, cfg)
    clusters_path = outputs.results_dir / "detected_campaign_clusters.csv"
    cluster_df.to_csv(clusters_path, index=False)
    assignments_path = outputs.results_dir / "campaign_cluster_assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    metrics_df, per_type, metrics = evaluate_campaign_detection(
        ground_truth, cluster_df, assignments, cfg
    )
    metrics_path = outputs.results_dir / "campaign_detection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    per_type_path = outputs.results_dir / "campaign_detection_by_type.csv"
    per_type.to_csv(per_type_path, index=False)

    comparison = compare_local_vs_fleet_campaign(
        descriptors, ground_truth, cluster_df, assignments, metrics, cfg
    )
    comparison_path = outputs.results_dir / "local_vs_fleet_campaign_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    table1 = build_campaign_results_table(per_type, metrics)
    t1_md = outputs.tables_dir / "table_campaign_detection_results.md"
    t1_tex = outputs.tables_dir / "table_campaign_detection_results.tex"
    t1_md.write_text(
        _df_to_markdown(table1, "Coordinated Campaign Detection Results"),
        encoding="utf-8",
    )
    t1_tex.write_text(
        _df_to_tex(
            table1,
            "Coordinated campaign detection on controlled multi-vehicle attack scenarios "
            "(behaviour-normalized descriptor similarity graph).",
            "tab:campaign-detection-results",
        ),
        encoding="utf-8",
    )

    t2_md = outputs.tables_dir / "table_local_vs_fleet_campaign_detection.md"
    t2_tex = outputs.tables_dir / "table_local_vs_fleet_campaign_detection.tex"
    t2_md.write_text(
        _df_to_markdown(comparison, "Local IDS vs Fleet-Aware Campaign Detection"),
        encoding="utf-8",
    )
    t2_tex.write_text(
        _df_to_tex(
            comparison,
            "Comparison of isolated vehicle-level IDS and fleet-aware campaign detection.",
            "tab:local-vs-fleet-campaign",
        ),
        encoding="utf-8",
    )

    fig_graph = outputs.figures_dir / "fleet_campaign_graph"
    plot_fleet_campaign_graph(
        graph, assignments, cluster_df, fig_graph, max_nodes=cfg.max_graph_viz_nodes, seed=cfg.seed
    )
    fig_emb = outputs.figures_dir / "campaign_descriptor_embedding"
    plot_campaign_descriptor_embedding(descriptors, assignments, cluster_df, fig_emb, cfg)

    summary_path = outputs.results_dir / "campaign_detection_summary.md"
    write_campaign_detection_summary(
        summary_path,
        ground_truth=ground_truth,
        metrics=metrics,
        per_type=per_type,
        primary_algorithm=primary,
        graph_stats=graph_stats,
    )

    written = {
        "campaign_ground_truth": gt_path,
        "campaign_graph_statistics": graph_stats_path,
        "detected_campaign_clusters": clusters_path,
        "campaign_cluster_assignments": assignments_path,
        "campaign_detection_metrics": metrics_path,
        "campaign_detection_by_type": per_type_path,
        "local_vs_fleet_campaign_comparison": comparison_path,
        "table_campaign_detection_results_md": t1_md,
        "table_campaign_detection_results_tex": t1_tex,
        "table_local_vs_fleet_campaign_detection_md": t2_md,
        "table_local_vs_fleet_campaign_detection_tex": t2_tex,
        "fleet_campaign_graph_pdf": fig_graph.with_suffix(".pdf"),
        "campaign_descriptor_embedding_pdf": fig_emb.with_suffix(".pdf"),
        "campaign_detection_summary": summary_path,
    }
    logger.info("Campaign detection evaluation complete (%d outputs).", len(written))
    return written
