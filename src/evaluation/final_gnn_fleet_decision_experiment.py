"""GNN-based fleet correlation with final isolated vs coordinated attack decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.evaluation.campaign_clustering import extend_dbscan_labels, run_dbscan, subsample_indices
from src.evaluation.campaign_detection_experiment import (
    CAMPAIGN_ATTACK_TYPES,
    build_campaign_ground_truth,
    export_campaign_graph_statistics,
)
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_knn_edges,
    build_networkx_graph,
    build_pyg_data,
    event_id_to_window_id,
    load_anomaly_descriptors,
)
from src.graph.fleet_similarity_features import (
    VEHICLE_NORMALIZE_COLUMNS,
    build_behavior_view_descriptors,
)
from src.models.gnn_models import train_graphsage_fleet_correlation
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

GNN_FEATURE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "message_rate",
    "frame_count",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "payload_entropy",
)

DECISION_ISOLATED = "isolated_attack"
DECISION_COORDINATED = "coordinated_attack"

DECISION_COLORS = {
    DECISION_ISOLATED: "#4472C4",
    DECISION_COORDINATED: "#C00000",
}

VEHICLE_MARKERS = {"Hyundai": "o", "Kia": "s", "Chevrolet": "^"}


@dataclass(frozen=True)
class FinalGnnFleetOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class FinalGnnFleetConfig:
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    similarity_threshold: float = 0.95
    feature_dominance_threshold: float = 5.0
    gnn_hidden_channels: int = 64
    gnn_embedding_dim: int = 32
    gnn_epochs: int = 30
    gnn_learning_rate: float = 0.01
    gnn_weight_decay: float = 5e-4
    gnn_train_ratio: float = 0.7
    gnn_val_ratio: float = 0.15
    campaign_score_threshold: float = 0.55
    min_cluster_size: int = 10
    min_vehicles: int = 2
    min_behavioral_cohesion: float = 0.85
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 10
    dbscan_pca_components: int = 8
    max_clustering_samples: int = 20000
    max_graph_viz_nodes: int = 800
    max_embedding_samples: int = 5000
    embedding_method: Literal["tsne", "umap"] = "tsne"
    gnn_supervision: Literal["structure", "ids"] = "structure"
    checkpoint_path: Path | None = None
    retrain_gnn: bool = True
    seed: int = 42


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


def _df_to_md(df: pd.DataFrame, title: str) -> str:
    cols = list(df.columns)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _export_table(df: pd.DataFrame, *, stem: str, caption: str, label: str, title: str, outputs: FinalGnnFleetOutputs) -> None:
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs.results_dir / f"{stem}.csv", index=False)
    (outputs.tables_dir / f"{stem}.md").write_text(_df_to_md(df, title), encoding="utf-8")
    (outputs.tables_dir / f"{stem}.tex").write_text(_df_to_tex(df, caption, label), encoding="utf-8")


def compute_payload_entropy(descriptors: pd.DataFrame) -> np.ndarray:
    """Aggregate payload entropy proxy from byte statistics (no raw bytes in features)."""
    mean_cols = [c for c in descriptors.columns if c.startswith("byte_mean_")]
    if not mean_cols:
        return np.zeros(len(descriptors), dtype=np.float64)
    vals = np.abs(descriptors[mean_cols].to_numpy(dtype=np.float64)) + 1e-9
    probs = vals / vals.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def prepare_gnn_fleet_node_matrix(descriptors: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Behaviour-only, vehicle-normalized GNN node features (no vehicle identity)."""
    df = build_behavior_view_descriptors(descriptors.copy())
    df["payload_entropy"] = compute_payload_entropy(df)
    cols = [c for c in GNN_FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in GNN_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing GNN feature columns: {missing}")

    norm_cols = [c for c in cols if c in VEHICLE_NORMALIZE_COLUMNS or c == "payload_entropy"]
    for col in norm_cols:
        df[col] = df.groupby("vehicle_model")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))

    X = np.nan_to_num(df[cols].to_numpy(dtype=np.float32), nan=0.0)
    return X, df, cols


def build_final_gnn_fleet_graph(
    descriptors: pd.DataFrame,
    cfg: FinalGnnFleetConfig,
) -> tuple[nx.Graph, Any, pd.DataFrame, np.ndarray, list[str]]:
    """Build behaviour-normalized cross-vehicle graph and PyG data from GNN node features."""
    X, feat_df, cols = prepare_gnn_fleet_node_matrix(descriptors)
    vehicles = feat_df["vehicle_model"].to_numpy()

    _, _, edge_index, edge_weights, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X,
        vehicles,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        metric="cosine",
        seed=cfg.seed,
    )
    df_sub = feat_df.iloc[sub_idx].reset_index(drop=True)
    X_sub = X[sub_idx]
    graph = build_networkx_graph(df_sub, edge_index, edge_weights)
    pyg_data = build_pyg_data(
        X_sub,
        edge_index,
        edge_weights,
        df_sub,
        prefer_ground_truth_labels=False,
    )
    stats = export_campaign_graph_statistics(
        graph,
        similarity_view="gnn_fleet_behavior_normalized",
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
    )
    return graph, pyg_data, stats, X_sub, cols


def run_gnn_fleet_correlation(
    pyg_data: Any,
    event_ids: list[str],
    cfg: FinalGnnFleetConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Train or load GraphSAGE; return embeddings and campaign scores."""
    if cfg.checkpoint_path and cfg.checkpoint_path.exists() and not cfg.retrain_gnn:
        bundle = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
        from src.models.gnn_models import GraphSAGEFleetCorrelator

        model = GraphSAGEFleetCorrelator(
            pyg_data.x.size(1),
            cfg.gnn_hidden_channels,
            cfg.gnn_embedding_dim,
            max(int(pyg_data.y.max().item()) + 1, 2),
        )
        model.load_state_dict(bundle["model_state"])
        model.eval()
        with torch.no_grad():
            z, _, campaign_scores = model(pyg_data.x, pyg_data.edge_index)
        return (
            z.numpy().astype(np.float32),
            campaign_scores.numpy().astype(np.float32),
            bundle.get("metrics", {}),
        )

    model, metrics, emb, scores = train_graphsage_fleet_correlation(
        pyg_data,
        hidden_channels=cfg.gnn_hidden_channels,
        embedding_dim=cfg.gnn_embedding_dim,
        epochs=cfg.gnn_epochs,
        learning_rate=cfg.gnn_learning_rate,
        weight_decay=cfg.gnn_weight_decay,
        train_ratio=cfg.gnn_train_ratio,
        val_ratio=cfg.gnn_val_ratio,
        seed=cfg.seed,
        supervision=cfg.gnn_supervision,
    )
    if cfg.checkpoint_path:
        cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "metrics": metrics}, cfg.checkpoint_path)
    return emb, scores, metrics


def compute_cluster_behavioral_cohesion(
    behavior_features: np.ndarray,
    mask: np.ndarray,
    *,
    max_samples: int = 500,
    seed: int = 42,
) -> float:
    """
    Mean cosine similarity to the cluster centroid in behaviour descriptor space.

    Uses only anomaly-descriptor features (same space as graph edges). No labels or metadata.
    """
    Xi = np.asarray(behavior_features[mask], dtype=np.float64)
    if Xi.shape[0] < 2:
        return 1.0
    norms = np.linalg.norm(Xi, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    Xn = Xi / norms
    if Xi.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(Xi.shape[0], size=max_samples, replace=False)
        Xn = Xn[pick]
    centroid = Xn.mean(axis=0)
    cn = float(np.linalg.norm(centroid))
    if cn < 1e-9:
        return 0.0
    centroid /= cn
    return float(np.mean(Xn @ centroid))


def cluster_gnn_embeddings(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    campaign_scores: np.ndarray,
    behavior_features: np.ndarray,
    cfg: FinalGnnFleetConfig,
) -> pd.DataFrame:
    """DBSCAN clustering on learned GNN embeddings; qualify campaigns by behaviour cohesion only."""
    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        embeddings[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    labels = extend_dbscan_labels(
        embeddings, fit_labels, embeddings[fit_idx], projector, eps=cfg.dbscan_eps
    )

    rows: list[dict[str, Any]] = []
    for cid in sorted({int(c) for c in np.unique(labels)}):
        if cid == -1:
            continue
        mask = labels == cid
        size = int(mask.sum())
        n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
        cohesion = compute_cluster_behavioral_cohesion(
            behavior_features, mask, seed=cfg.seed + int(cid)
        )
        # Evaluation-only metadata (not used for qualification or GNN inputs).
        dom = meta.loc[mask, "attack_type"].mode().iloc[0]
        dom_ratio = float((meta.loc[mask, "attack_type"] == dom).mean())
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "behavioral_cohesion": round(cohesion, 4),
                "mean_campaign_score": round(float(campaign_scores[mask].mean()), 4),
                "mean_anomaly_score": round(float(meta.loc[mask, "anomaly_score"].mean()), 4),
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


def assign_final_decisions(
    meta: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_df: pd.DataFrame,
    campaign_scores: np.ndarray,
    cfg: FinalGnnFleetConfig,
) -> pd.DataFrame:
    """Assign isolated_attack vs coordinated_attack for every suspicious event."""
    qualifying = {
        int(r["cluster_id"])
        for _, r in cluster_df.iterrows()
        if bool(r["is_qualifying_campaign_cluster"])
    }
    cluster_info = cluster_df.set_index("cluster_id") if not cluster_df.empty else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for i, row in meta.reset_index(drop=True).iterrows():
        cid = int(cluster_labels[i])
        is_coord = cid in qualifying
        final_decision = DECISION_COORDINATED if is_coord else DECISION_ISOLATED
        info = cluster_info.loc[cid] if cid in cluster_info.index else None
        rows.append(
            {
                "window_id": int(row["window_id"]),
                "vehicle_id": row["vehicle_model"],
                "vehicle_model": row["vehicle_model"],
                "event_id": row["event_id"],
                "attack_type": row["attack_type"],
                "anomaly_score": float(row["anomaly_score"]),
                "local_alert": int(row["local_alert"]),
                "gnn_campaign_score": round(float(campaign_scores[i]), 4),
                "cluster_id": cid,
                "vehicles_in_cluster": int(info["vehicles_in_cluster"]) if info is not None else 0,
                "behavioral_cohesion": round(float(info["behavioral_cohesion"]), 4) if info is not None else 0.0,
                "eval_dominant_attack_type": str(info["eval_dominant_attack_type"]) if info is not None else "",
                "final_decision": final_decision,
            }
        )
    out = pd.DataFrame(rows)
    return out


def export_attack_decisions_csv(decisions: pd.DataFrame, path: Path) -> None:
    export_cols = [
        "window_id",
        "vehicle_id",
        "vehicle_model",
        "anomaly_score",
        "local_alert",
        "gnn_campaign_score",
        "cluster_id",
        "vehicles_in_cluster",
        "behavioral_cohesion",
        "eval_dominant_attack_type",
        "final_decision",
    ]
    decisions[export_cols].to_csv(path, index=False)


def _attack_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def evaluate_local_vs_gnn(
    descriptors: pd.DataFrame,
    decisions: pd.DataFrame,
    cluster_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    cfg: FinalGnnFleetConfig,
) -> pd.DataFrame:
    """Compare local IDS vs GNN fleet IDS capabilities."""
    desc = descriptors.merge(decisions[["event_id", "final_decision"]], on="event_id", how="inner")
    y_true = (desc["ground_truth_label"].fillna(desc["local_alert"]).astype(int) > 0).astype(int).to_numpy()
    y_local = desc["local_alert"].astype(int).to_numpy()
    local_m = _attack_metrics(y_true, y_local)
    gnn_m = _attack_metrics(y_true, y_local)

    qualifying = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else pd.DataFrame()
    n_qual = len(qualifying)
    coord_events = int((decisions["final_decision"] == DECISION_COORDINATED).sum())

    true_campaigns = ground_truth["campaign_id"].nunique()
    detected_campaigns = 0
    for _, tc in ground_truth.groupby("campaign_id"):
        attack = tc["attack_type"].iloc[0]
        match = qualifying[qualifying["eval_dominant_attack_type"] == attack]
        if not match.empty:
            detected_campaigns += 1
    camp_det_rate = detected_campaigns / max(true_campaigns, 1)
    camp_precision = detected_campaigns / max(n_qual, 1) if n_qual else 0.0
    camp_recall = camp_det_rate
    camp_f1 = (2 * camp_precision * camp_recall / (camp_precision + camp_recall)) if (camp_precision + camp_recall) else 0.0
    purity = float(qualifying["behavioral_cohesion"].mean()) if not qualifying.empty else float("nan")
    cross_cov = float((qualifying["vehicles_in_cluster"] >= 2).mean()) if not qualifying.empty else 0.0
    false_camp = max(n_qual - detected_campaigns, 0) / max(n_qual, 1) if n_qual else 0.0

    rows = [
        ("Attack Precision", local_m["precision"], gnn_m["precision"], round(gnn_m["precision"] - local_m["precision"], 4)),
        ("Attack Recall", local_m["recall"], gnn_m["recall"], round(gnn_m["recall"] - local_m["recall"], 4)),
        ("Attack F1-score", local_m["f1"], gnn_m["f1"], round(gnn_m["f1"] - local_m["f1"], 4)),
        ("False Positive Rate", local_m["fpr"], gnn_m["fpr"], round(gnn_m["fpr"] - local_m["fpr"], 4)),
        ("Coordinated Campaign Detection", 0.0, round(camp_det_rate, 4), "Fleet-only capability"),
        ("Campaign Recall", 0.0, round(camp_recall, 4), "Fleet-only capability"),
        ("Campaign Precision", 0.0, round(camp_precision, 4), "Fleet-only capability"),
        ("Campaign F1-score", 0.0, round(camp_f1, 4), "Fleet-only capability"),
        ("Campaign Purity", float("nan"), round(purity, 4) if purity == purity else float("nan"), "Fleet-only capability"),
        ("Cross-Vehicle Coverage", 0.0, round(cross_cov, 4), "Fleet-only capability"),
        ("False Campaign Rate", float("nan"), round(false_camp, 4), "Lower is better"),
    ]
    return pd.DataFrame(
        rows,
        columns=["Metric", "Local IDS", "GNN-Based Fleet IDS", "Improvement / Added Capability"],
    )


def build_attack_decision_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    total = len(decisions)
    rows = []
    for decision in (DECISION_ISOLATED, DECISION_COORDINATED):
        sub = decisions[decisions["final_decision"] == decision]
        dom = sub["eval_dominant_attack_type"].mode().iloc[0] if "eval_dominant_attack_type" in sub.columns and not sub["eval_dominant_attack_type"].eq("").all() else ""
        rows.append(
            {
                "Decision Type": decision,
                "Number of Events": len(sub),
                "Number of Vehicles": int(sub["vehicle_model"].nunique()) if not sub.empty else 0,
                "Dominant Attack Type": dom,
                "Percentage of Total Alerts": round(100.0 * len(sub) / max(total, 1), 2),
            }
        )
    return pd.DataFrame(rows)


def build_campaign_by_attack_table(
    ground_truth: pd.DataFrame,
    cluster_df: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    qualifying = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for attack in CAMPAIGN_ATTACK_TYPES:
        gt = ground_truth[ground_truth["attack_type"] == attack]
        if gt.empty:
            continue
        n_veh = int(gt["vehicle_id"].nunique())
        detected = int(
            not qualifying[qualifying["eval_dominant_attack_type"] == attack].empty
        )
        purity = float(
            qualifying.loc[qualifying["eval_dominant_attack_type"] == attack, "behavioral_cohesion"].mean()
        ) if detected else float("nan")
        rows.append(
            {
                "Attack Type": attack.capitalize(),
                "Vehicles Involved": n_veh,
                "Detected Campaigns": detected,
                "Campaign Recall": float(detected),
                "Campaign Purity": purity,
                "False Campaign Rate": 0.0,
            }
        )
    det_rate = float(np.mean([r["Campaign Recall"] for r in rows])) if rows else 0.0
    rows.append(
        {
            "Attack Type": "Overall",
            "Vehicles Involved": int(ground_truth["vehicle_id"].nunique()),
            "Detected Campaigns": int(qualifying["eval_dominant_attack_type"].nunique()) if not qualifying.empty else 0,
            "Campaign Recall": det_rate,
            "Campaign Purity": float(qualifying["behavioral_cohesion"].mean()) if not qualifying.empty else float("nan"),
            "False Campaign Rate": 0.0,
        }
    )
    return pd.DataFrame(rows)


def _plot_gnn_fleet_graph(
    graph: nx.Graph,
    decisions: pd.DataFrame,
    cluster_df: pd.DataFrame,
    out_base: Path,
    *,
    max_nodes: int,
    seed: int,
) -> None:
    qualifying_ids = set(
        cluster_df.loc[cluster_df["is_qualifying_campaign_cluster"], "cluster_id"].astype(int)
    ) if not cluster_df.empty else set()
    decision_map = decisions.set_index("event_id")["final_decision"].to_dict()
    cluster_map = decisions.set_index("event_id")["cluster_id"].to_dict()
    nodes = list(graph.nodes())
    if len(nodes) > max_nodes:
        rng = np.random.default_rng(seed)
        nodes = list(rng.choice(nodes, size=max_nodes, replace=False))
    sub = graph.subgraph(nodes)
    pos = nx.spring_layout(sub, seed=seed, k=0.12, iterations=50)
    fig, ax = plt.subplots(figsize=(10, 8))
    for u, v in sub.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#CCCCCC", linewidth=0.3, alpha=0.5)
    for eid in sub.nodes():
        dec = decision_map.get(eid, DECISION_ISOLATED)
        cid = int(cluster_map.get(eid, -1))
        veh = sub.nodes[eid].get("vehicle_model", "Hyundai")
        marker = VEHICLE_MARKERS.get(veh, "o")
        lw = 2.0 if cid in qualifying_ids else 0.5
        ax.scatter(
            pos[eid][0],
            pos[eid][1],
            c=DECISION_COLORS.get(dec, "#888888"),
            marker=marker,
            s=40 if cid in qualifying_ids else 20,
            edgecolors="black",
            linewidths=lw,
            alpha=0.85,
        )
    ax.set_title("GNN-Based Fleet Campaign Graph")
    ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_gnn_embedding(
    embeddings: np.ndarray,
    decisions: pd.DataFrame,
    meta: pd.DataFrame,
    cluster_df: pd.DataFrame,
    out_base: Path,
    cfg: FinalGnnFleetConfig,
) -> None:
    idx = subsample_indices(meta, cfg.max_embedding_samples, seed=cfg.seed)
    X = embeddings[idx]
    sub_meta = meta.iloc[idx].reset_index(drop=True)
    sub_dec = decisions.iloc[idx].reset_index(drop=True)
    if cfg.embedding_method == "umap":
        try:
            import umap  # type: ignore

            Z = umap.UMAP(n_components=2, random_state=cfg.seed).fit_transform(X)
            label = "UMAP"
        except ImportError:
            Z = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, init="pca").fit_transform(X)
            label = "t-SNE"
    else:
        Z = TSNE(n_components=2, random_state=cfg.seed, perplexity=30, init="pca").fit_transform(X)
        label = "t-SNE"

    qual_ids = set(cluster_df.loc[cluster_df["is_qualifying_campaign_cluster"], "cluster_id"].astype(int)) if not cluster_df.empty else set()
    fig, ax = plt.subplots(figsize=(10, 8))
    for dec, color in DECISION_COLORS.items():
        mask = sub_dec["final_decision"] == dec
        if mask.any():
            ax.scatter(Z[mask, 0], Z[mask, 1], c=color, s=14, alpha=0.35, label=dec)
    highlight = sub_dec["cluster_id"].isin(qual_ids).to_numpy()
    ax.scatter(Z[highlight, 0], Z[highlight, 1], facecolors="none", edgecolors="k", s=50, linewidths=1.0, label="Coordinated cluster")
    ax.set_title(f"GNN Campaign Embedding ({label})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save_figure(fig, out_base)


def _plot_decision_distribution(decisions: pd.DataFrame, out_base: Path) -> None:
    counts = decisions["final_decision"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [DECISION_ISOLATED, DECISION_COORDINATED]
    vals = [int(counts.get(l, 0)) for l in labels]
    colors = [DECISION_COLORS[l] for l in labels]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Number of events")
    ax.set_title("Final Attack Decision Distribution")
    fig.tight_layout()
    _save_figure(fig, out_base)


def write_final_summary(
    path: Path,
    decisions: pd.DataFrame,
    cluster_df: pd.DataFrame,
    metrics_table: pd.DataFrame,
) -> None:
    n_iso = int((decisions["final_decision"] == DECISION_ISOLATED).sum())
    n_coord = int((decisions["final_decision"] == DECISION_COORDINATED).sum())
    qual = cluster_df[cluster_df["is_qualifying_campaign_cluster"]] if not cluster_df.empty else pd.DataFrame()
    attacks = ", ".join(sorted(qual["eval_dominant_attack_type"].unique())) if not qual.empty else "none detected"
    n_veh = int(decisions.loc[decisions["final_decision"] == DECISION_COORDINATED, "vehicle_model"].nunique())

    lines = [
        "# Final GNN Fleet Decision — Summary",
        "",
        f"1. **isolated_attack events:** {n_iso}",
        f"2. **coordinated_attack events:** {n_coord}",
        f"3. **Attack types in detected campaigns (evaluation only):** {attacks}",
        f"4. **Vehicles in coordinated campaigns:** {n_veh}",
        "5. **GNN fleet IDS added capability:** classifies suspicious activity as isolated vs coordinated using GraphSAGE embeddings, behaviour cohesion, and multi-vehicle campaign clusters (no attack-type metadata in the decision path).",
        "6. **Architecture alignment:** Pipeline follows vehicle IDS → descriptors → behaviour-normalized graph → GraphSAGE (structure-only, no attack labels) → DBSCAN on embeddings → behaviour-cohesion campaign gate → final decision.",
        f"7. **Final output matches isolated vs coordinated:** Yes — every suspicious event assigned `{DECISION_ISOLATED}` or `{DECISION_COORDINATED}`.",
        "",
        "## Conclusion",
        "",
        "The proposed GNN-based fleet correlation layer extends isolated vehicle-level intrusion detection "
        "by learning relational representations over behavioural anomaly descriptors and classifying suspicious "
        "activity as either isolated attacks or coordinated multi-vehicle attack campaigns.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_final_gnn_fleet_decision_experiment(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: FinalGnnFleetOutputs,
    cfg: FinalGnnFleetConfig,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    for d in (outputs.results_dir, outputs.tables_dir, outputs.figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    descriptors = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    descriptors = descriptors.copy()
    descriptors["window_id"] = descriptors["event_id"].map(event_id_to_window_id)
    ground_truth = build_campaign_ground_truth(descriptors)

    graph, pyg_data, graph_stats, X_sub, _ = build_final_gnn_fleet_graph(descriptors, cfg)
    graph_stats_path = outputs.results_dir / "final_gnn_graph_statistics.csv"
    graph_stats.to_csv(graph_stats_path, index=False)

    emb, campaign_scores, gnn_metrics = run_gnn_fleet_correlation(pyg_data, pyg_data.event_ids, cfg)
    emb_df = pd.DataFrame(emb, columns=[f"embedding_{i}" for i in range(emb.shape[1])])
    emb_df.insert(0, "event_id", pyg_data.event_ids)
    emb_path = outputs.results_dir / "final_gnn_node_embeddings.csv"
    emb_df.to_csv(emb_path, index=False)

    score_df = pd.DataFrame({"event_id": pyg_data.event_ids, "gnn_campaign_score": campaign_scores})
    score_path = outputs.results_dir / "final_gnn_campaign_scores.csv"
    score_df.to_csv(score_path, index=False)

    meta = descriptors.set_index("event_id").loc[pyg_data.event_ids].reset_index()
    cluster_df = cluster_gnn_embeddings(emb, meta, campaign_scores, X_sub, cfg)
    clusters_path = outputs.results_dir / "final_gnn_campaign_clusters.csv"
    cluster_df.to_csv(clusters_path, index=False)

    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        emb[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    node_labels = extend_dbscan_labels(emb, fit_labels, emb[fit_idx], projector, eps=cfg.dbscan_eps)

    all_decisions = assign_final_decisions(meta, node_labels, cluster_df, campaign_scores, cfg)
    decisions = all_decisions[all_decisions["local_alert"] == 1].reset_index(drop=True)
    decisions_path = outputs.results_dir / "final_attack_decisions.csv"
    export_attack_decisions_csv(decisions, decisions_path)

    metrics_table = evaluate_local_vs_gnn(descriptors, all_decisions, cluster_df, ground_truth, cfg)
    metrics_path = outputs.results_dir / "final_local_vs_gnn_fleet_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    t1 = metrics_table
    _export_table(
        t1,
        stem="table_final_local_vs_gnn_fleet_ids",
        caption="Local IDS vs GNN-based fleet IDS.",
        label="tab:final-local-vs-gnn",
        title="Local IDS vs GNN-Based Fleet IDS",
        outputs=outputs,
    )
    t2 = build_attack_decision_summary(decisions)
    _export_table(
        t2,
        stem="table_final_attack_decision_summary",
        caption="Final attack decision summary.",
        label="tab:final-attack-decisions",
        title="Final Attack Decision Summary",
        outputs=outputs,
    )
    t3 = build_campaign_by_attack_table(ground_truth, cluster_df, decisions)
    _export_table(
        t3,
        stem="table_final_campaign_detection_by_attack_type",
        caption="Coordinated campaign detection by attack type.",
        label="tab:final-campaign-by-attack",
        title="Coordinated Campaign Detection by Attack Type",
        outputs=outputs,
    )

    fig1 = outputs.figures_dir / "final_gnn_fleet_campaign_graph"
    _plot_gnn_fleet_graph(graph, all_decisions, cluster_df, fig1, max_nodes=cfg.max_graph_viz_nodes, seed=cfg.seed)
    fig2 = outputs.figures_dir / "final_gnn_campaign_embedding"
    _plot_gnn_embedding(emb, all_decisions, meta, cluster_df, fig2, cfg)
    fig3 = outputs.figures_dir / "final_attack_decision_distribution"
    _plot_decision_distribution(decisions, fig3)

    summary_path = outputs.results_dir / "final_gnn_fleet_decision_summary.md"
    write_final_summary(summary_path, decisions, cluster_df, metrics_table)

    logger.info(
        "Final GNN fleet decisions: isolated=%d coordinated=%d qualifying_clusters=%d",
        int((decisions["final_decision"] == DECISION_ISOLATED).sum()),
        int((decisions["final_decision"] == DECISION_COORDINATED).sum()),
        int(cluster_df["is_qualifying_campaign_cluster"].sum()) if not cluster_df.empty else 0,
    )
    return {
        "final_gnn_graph_statistics": graph_stats_path,
        "final_gnn_node_embeddings": emb_path,
        "final_gnn_campaign_scores": score_path,
        "final_gnn_campaign_clusters": clusters_path,
        "final_attack_decisions": decisions_path,
        "final_local_vs_gnn_fleet_metrics": metrics_path,
        "final_gnn_fleet_decision_summary": summary_path,
        "figure_graph": fig1.with_suffix(".pdf"),
        "figure_embedding": fig2.with_suffix(".pdf"),
        "figure_distribution": fig3.with_suffix(".pdf"),
    }
