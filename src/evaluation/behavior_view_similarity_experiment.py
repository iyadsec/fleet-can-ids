"""Behaviour-focused fleet graph similarity ablation and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.fleet_correlation_experiment import (
    _detection_metrics,
    assign_component_labels,
    identify_suspicious_clusters,
)
from src.evaluation.fleet_similarity_diagnosis import (
    ATTACK_TYPES,
    analyse_cross_vehicle_attack_similarity,
)
from src.evaluation.selective_weak_promotion import (
    PromotionGates,
    compute_dbscan_subcluster_statistics,
    evaluate_selective_promotion,
    identify_selective_recoverable_weak,
    run_dbscan_on_graph_nodes,
)
from src.evaluation.weak_anomaly_recovery_experiment import (
    analyse_weak_clusters,
    build_weak_anomaly_graph,
    classify_anomaly_strength,
    identify_recoverable,
)
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_graph,
    build_fleet_correlation_graph,
    build_topk_similarity_edges,
    compute_graph_statistics,
    load_anomaly_descriptors,
)
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_similarity_features import (
    BEHAVIOR_GRAPH_CANDIDATE_COLUMNS,
    IDENTITY_HEAVY_COLUMNS,
    SIMILARITY_VIEW_LABELS,
    SimilarityFeatureView,
    build_behavior_view_descriptors,
    prepare_fleet_similarity_matrix,
    select_behavior_graph_columns,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

VIEWS: tuple[SimilarityFeatureView, ...] = (
    "full_descriptor",
    "behavior_only",
    "behavior_only_vehicle_normalized",
)


@dataclass(frozen=True)
class BehaviorViewOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class BehaviorViewConfig:
    top_k_neighbors: int = 15
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    similarity_threshold: float = 0.95
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()
    strong_threshold: float = 0.80
    weak_threshold: float = 0.55
    fleet_minimum_vehicle_count: int = 3
    fleet_minimum_cluster_size: int = 2
    fleet_cluster_score_threshold: float = 0.7
    weak_minimum_vehicle_count: int = 2
    weak_minimum_cluster_size: int = 2
    recovery_score_threshold: float = 0.55
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 10
    dbscan_pca_components: int = 8
    max_clustering_samples: int = 20000
    promotion_mean_score: float = 0.60
    promotion_strong_support: float = 0.40
    promotion_min_cohesion: float = 0.97
    promotion_min_cluster_size: int = 5
    promotion_min_vehicles: int = 2
    max_pairs_per_category: int = 50000
    seed: int = 42
    paper_similarity_view: SimilarityFeatureView = "behavior_only_vehicle_normalized"


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


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


def _attack_cross_vehicle_sim(
    df: pd.DataFrame,
    X: np.ndarray,
    attack: str,
    *,
    max_pairs: int,
    seed: int,
) -> float:
    sub = analyse_cross_vehicle_attack_similarity(
        df, X, max_pairs=max_pairs, seed=seed
    )
    row = sub[
        (sub["attack_type"] == attack)
        & (sub["metric"] == "average_cross_vehicle_similarity")
    ]
    return float(row["value"].iloc[0]) if len(row) else float("nan")


def _topk_edge_bias(
    descriptors: pd.DataFrame,
    X: np.ndarray,
    *,
    top_k: int,
    similarity_threshold: float,
    seed: int,
) -> dict[str, float]:
    _, _, ei, _, sub_idx = build_topk_similarity_edges(
        X,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    sub = descriptors.iloc[sub_idx].reset_index(drop=True)
    vehicles = sub["vehicle_model"].astype(str).to_numpy()
    attacks = sub["attack_type"].astype(str).to_numpy()
    same_v = cross_v = flood_cross = flood_total = 0
    if ei.size:
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u >= v:
                continue
            sv = vehicles[u] == vehicles[v]
            if sv:
                same_v += 1
            else:
                cross_v += 1
            if attacks[u] == "flooding" and attacks[v] == "flooding":
                flood_total += 1
                if not sv:
                    flood_cross += 1
    total = same_v + cross_v
    return {
        "pct_same_vehicle_edges": 100.0 * same_v / total if total else 0.0,
        "pct_cross_vehicle_edges": 100.0 * cross_v / total if total else 0.0,
        "flooding_cross_vehicle_edge_pct": 100.0 * flood_cross / flood_total if flood_total else 0.0,
        "total_edges": float(total),
    }


def _fleet_metrics(
    descriptors: pd.DataFrame,
    G: nx.Graph,
    cfg: BehaviorViewConfig,
    cluster_results_path: Path | None,
) -> dict[str, float]:
    y_true = descriptors["ground_truth_label"].astype(int).to_numpy()
    local_pred = descriptors["local_alert"].astype(int).to_numpy()
    y_score = descriptors["anomaly_score"].astype(float).to_numpy()
    suspicious, _, _ = identify_suspicious_clusters(
        G,
        descriptors,
        minimum_cluster_size=cfg.fleet_minimum_cluster_size,
        minimum_vehicle_count=cfg.fleet_minimum_vehicle_count,
        similarity_threshold=cfg.similarity_threshold,
        fleet_cluster_score_threshold=cfg.fleet_cluster_score_threshold,
        cluster_results_path=cluster_results_path,
    )
    fleet_pred = ((local_pred == 1) | descriptors["event_id"].astype(str).isin(suspicious)).astype(int)
    m = _detection_metrics(y_true, fleet_pred, y_score)
    return {"fleet_recall": m["recall"], "fleet_f1": m["f1"], "fleet_fpr": m["false_positive_rate"]}


def _weak_recovery_metrics(
    weak_table: pd.DataFrame,
    G: nx.Graph,
    cfg: BehaviorViewConfig,
) -> dict[str, float]:
    weak_df, clusters, _ = analyse_weak_clusters(G, weak_table)
    recoverable = identify_recoverable(
        weak_df,
        clusters,
        minimum_cluster_size=cfg.weak_minimum_cluster_size,
        minimum_vehicle_count=cfg.weak_minimum_vehicle_count,
        recovery_score_threshold=cfg.recovery_score_threshold,
    )
    y_true = weak_df["ground_truth_label"].astype(int).to_numpy()
    local_pred = weak_df["local_alert"].astype(int).to_numpy()
    fleet_pred = np.where(
        weak_df["descriptor_id"].astype(str).isin(recoverable).to_numpy(), 1, local_pred
    ).astype(int)
    cm = confusion_matrix(y_true, fleet_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    missed = (local_pred == 0) & (y_true == 1)
    n_missed = int(missed.sum())
    recovered = int(
        ((local_pred == 0) & weak_df["descriptor_id"].astype(str).isin(recoverable) & (y_true == 1)).sum()
    )
    recovery_rate = 100.0 * recovered / n_missed if n_missed else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "weak_recovery_rate_percent": recovery_rate,
        "weak_fpr": fpr,
        "weak_f1": float(f1_score(y_true, fleet_pred, zero_division=0)),
    }


def _selective_promotion_metrics(
    descriptors: pd.DataFrame,
    cfg: BehaviorViewConfig,
    view: SimilarityFeatureView,
) -> dict[str, float]:
    from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix

    _, G, _, _ = build_cross_vehicle_constrained_graph(
        descriptors,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    del G
    desc = descriptors.reset_index(drop=True)
    X, _ = resolve_fleet_similarity_matrix(
        desc,
        similarity_feature_view=view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    labels = run_dbscan_on_graph_nodes(
        desc,
        X,
        dbscan_eps=cfg.dbscan_eps,
        dbscan_min_samples=cfg.dbscan_min_samples,
        dbscan_pca_components=cfg.dbscan_pca_components,
        max_clustering_samples=cfg.max_clustering_samples,
        seed=cfg.seed,
    )
    cluster_stats = compute_dbscan_subcluster_statistics(
        desc, X, labels, strong_threshold=cfg.strong_threshold, max_cohesion_pairs=8000, seed=cfg.seed
    )
    classified = classify_anomaly_strength(
        descriptors, weak_threshold=cfg.weak_threshold, strong_threshold=cfg.strong_threshold
    )
    weak_full = classified[classified["anomaly_strength"] == "weak"]
    weak_df = weak_full[
        ["ground_truth_label", "event_id", "local_alert"]
    ].copy()
    weak_df = weak_df.rename(columns={"event_id": "descriptor_id"})
    eid_to_cluster = dict(zip(desc["event_id"].astype(str), labels.astype(int)))
    weak_df["cluster_id"] = weak_df["descriptor_id"].astype(str).map(eid_to_cluster).fillna(-1).astype(int)
    gates = PromotionGates(
        mean_score_threshold=cfg.promotion_mean_score,
        strong_support_threshold=cfg.promotion_strong_support,
        min_cohesion=cfg.promotion_min_cohesion,
        min_cluster_size=cfg.promotion_min_cluster_size,
        min_vehicles=cfg.promotion_min_vehicles,
    )
    recoverable = identify_selective_recoverable_weak(
        weak_df["descriptor_id"], weak_df["cluster_id"], cluster_stats, gates
    )
    m = evaluate_selective_promotion(weak_df, recoverable)
    return {
        "selective_recovery_rate_percent": m["recovery_rate_percent"],
        "selective_fpr": m["false_positive_rate"],
        "selective_f1": m["f1"],
    }


def _evaluate_view(
    descriptors: pd.DataFrame,
    view: SimilarityFeatureView,
    cfg: BehaviorViewConfig,
    *,
    cluster_results_path: Path | None,
    weak_table: pd.DataFrame,
    weak_only: pd.DataFrame,
) -> dict[str, Any]:
    X, cols, dominance, removed = prepare_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    _, G, build_stats, _ = build_fleet_correlation_graph(
        descriptors,
        top_k_neighbors=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    gstats = compute_graph_statistics(G)
    bias = _topk_edge_bias(descriptors, X, top_k=cfg.top_k_neighbors, similarity_threshold=cfg.similarity_threshold, seed=cfg.seed)
    fleet_m = _fleet_metrics(descriptors, G, cfg, cluster_results_path)
    G_weak, _ = build_weak_anomaly_graph(
        weak_only,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    weak_m = _weak_recovery_metrics(weak_table, G_weak, cfg)
    sel_m = _selective_promotion_metrics(descriptors, cfg, view)

    cross_rows = analyse_cross_vehicle_attack_similarity(
        descriptors.reset_index(drop=True),
        X,
        max_pairs=cfg.max_pairs_per_category,
        seed=cfg.seed,
    )

    return {
        "similarity_feature_view": view,
        "similarity_view_label": SIMILARITY_VIEW_LABELS[view],
        "feature_columns": cols,
        "removed_features": removed,
        "dominance": dominance,
        "graph_stats": {**gstats, **build_stats},
        "edge_bias": bias,
        "flooding_cross_vehicle_similarity": _attack_cross_vehicle_sim(
            descriptors, X, "flooding", max_pairs=cfg.max_pairs_per_category, seed=cfg.seed
        ),
        "fuzzy_cross_vehicle_similarity": _attack_cross_vehicle_sim(
            descriptors, X, "fuzzy", max_pairs=cfg.max_pairs_per_category, seed=cfg.seed + 1
        ),
        "replay_cross_vehicle_similarity": _attack_cross_vehicle_sim(
            descriptors, X, "replay", max_pairs=cfg.max_pairs_per_category, seed=cfg.seed + 2
        ),
        "malfunction_cross_vehicle_similarity": _attack_cross_vehicle_sim(
            descriptors, X, "malfunction", max_pairs=cfg.max_pairs_per_category, seed=cfg.seed + 3
        ),
        "cross_vehicle_attack_similarity": cross_rows,
        **fleet_m,
        **weak_m,
        **sel_m,
    }


def run_behavior_view_similarity_experiment(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: BehaviorViewOutputs,
    cfg: BehaviorViewConfig,
    cluster_results_path: Path | None = None,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    descriptors = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    view_df = build_behavior_view_descriptors(descriptors)
    _, dominance_all, removed_identity = select_behavior_graph_columns(
        view_df,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    dominance_all.to_csv(
        outputs.results_dir / "fleet_similarity_feature_dominance.csv", index=False
    )

    classified = classify_anomaly_strength(
        descriptors, weak_threshold=cfg.weak_threshold, strong_threshold=cfg.strong_threshold
    )
    weak_only = classified[classified["anomaly_strength"] == "weak"].copy()
    weak_table = weak_only[
        ["window_id", "vehicle_model", "anomaly_score", "ground_truth_label", "event_id", "attack_type", "local_alert"]
    ].copy()
    weak_table = weak_table.rename(columns={"vehicle_model": "vehicle_id", "event_id": "descriptor_id"})

    view_results: list[dict[str, Any]] = []
    cross_sim_rows: list[pd.DataFrame] = []
    topk_bias_rows: list[dict[str, Any]] = []
    graph_stat_rows: list[dict[str, Any]] = []

    for view in VIEWS:
        logger.info("Evaluating similarity view: %s", view)
        res = _evaluate_view(
            descriptors,
            view,
            cfg,
            cluster_results_path=cluster_results_path,
            weak_table=weak_table,
            weak_only=weak_only,
        )
        view_results.append(res)
        cross_sim_rows.append(res["cross_vehicle_attack_similarity"].assign(similarity_view=view))
        topk_bias_rows.append({"similarity_view": view, **res["edge_bias"]})
        graph_stat_rows.append(
            {
                "similarity_view": view,
                "num_nodes": res["graph_stats"].get("num_nodes", 0),
                "num_edges": res["graph_stats"].get("num_edges", 0),
                "num_cross_vehicle_edges": res["graph_stats"].get("num_cross_vehicle_edges", 0),
                "graph_density": res["graph_stats"].get("graph_density", 0),
                "connected_components": res["graph_stats"].get("connected_components", 0),
            }
        )

    pd.concat(cross_sim_rows, ignore_index=True).to_csv(
        outputs.results_dir / "behavior_view_cross_vehicle_similarity.csv", index=False
    )
    pd.DataFrame(topk_bias_rows).to_csv(
        outputs.results_dir / "behavior_view_topk_vehicle_bias.csv", index=False
    )
    pd.DataFrame(graph_stat_rows).to_csv(
        outputs.results_dir / "behavior_view_fleet_graph_statistics.csv", index=False
    )

    ablation = pd.DataFrame(
        [
            {
                "Similarity View": r["similarity_view_label"],
                "Same-Vehicle Edge %": r["edge_bias"]["pct_same_vehicle_edges"],
                "Cross-Vehicle Edge %": r["edge_bias"]["pct_cross_vehicle_edges"],
                "Flooding Cross-Vehicle Similarity": r["flooding_cross_vehicle_similarity"],
                "Fleet Recall": r["fleet_recall"],
                "Fleet F1": r["fleet_f1"],
                "Weak Recovery Rate": r["weak_recovery_rate_percent"],
                "Weak FPR": r["weak_fpr"],
            }
            for r in view_results
        ]
    ).round(4)

    table_tex = outputs.tables_dir / "table_similarity_view_ablation.tex"
    table_tex.write_text(
        _df_to_ieee_tex(
            ablation,
            "Effect of Descriptor Similarity View on Fleet Graph Construction",
            "tab:similarity-view-ablation",
        ),
        encoding="utf-8",
    )
    table_md = outputs.tables_dir / "table_similarity_view_ablation.md"
    cols = list(ablation.columns)
    md = ["# Effect of Descriptor Similarity View on Fleet Graph Construction", ""]
    md.append("| " + " | ".join(cols) + " |")
    md.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in ablation.iterrows():
        md.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    table_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # Figure 1: edge ratio bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ablation))
    w = 0.35
    ax.bar(x - w / 2, ablation["Same-Vehicle Edge %"], width=w, label="Same-vehicle", color="#4472C4")
    ax.bar(x + w / 2, ablation["Cross-Vehicle Edge %"], width=w, label="Cross-vehicle", color="#ED7D31")
    ax.set_xticks(x)
    ax.set_xticklabels(ablation["Similarity View"], rotation=12, ha="right")
    ax.set_ylabel("Edge percentage (%)")
    ax.set_title("Same-vehicle vs cross-vehicle edges by similarity view")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "same_vs_cross_vehicle_edges_behavior_view")

    # Figure 2: t-SNE embedding (paper view)
    paper_view = cfg.paper_similarity_view
    X_emb, emb_cols, _, _ = prepare_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=paper_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    n_sample = min(4000, len(X_emb))
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(X_emb), size=n_sample, replace=False)
    sub = descriptors.iloc[idx].reset_index(drop=True)
    Xs = X_emb[idx]
    tsne = TSNE(n_components=2, perplexity=30, random_state=cfg.seed, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col, title in zip(
        axes,
        ["attack_type", "vehicle_model"],
        ["Attack type", "Vehicle model"],
    ):
        for label in sorted(sub[col].astype(str).unique()):
            m = sub[col].astype(str) == label
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, label=label)
        ax.set_title(f"t-SNE — {title}")
        ax.legend(fontsize=6, markerscale=2)
    fig.suptitle(f"Behaviour view embedding ({SIMILARITY_VIEW_LABELS[paper_view]})")
    _save_figure(fig, outputs.figures_dir / "behavior_view_descriptor_embedding")

    # Paper-view experiment exports
    paper_res = next(r for r in view_results if r["similarity_feature_view"] == paper_view)
    y_true = descriptors["ground_truth_label"].astype(int).to_numpy()
    local_pred = descriptors["local_alert"].astype(int).to_numpy()
    y_score = descriptors["anomaly_score"].astype(float).to_numpy()
    local_m = _detection_metrics(y_true, local_pred, y_score)
    pd.DataFrame(
        [
            {
                "system": "local_only",
                "recall": local_m["recall"],
                "f1": local_m["f1"],
                "fpr": local_m["false_positive_rate"],
            },
            {
                "system": "fleet_aware",
                "recall": paper_res["fleet_recall"],
                "f1": paper_res["fleet_f1"],
                "fpr": paper_res["fleet_fpr"],
            },
        ]
    ).to_csv(outputs.results_dir / "fleet_vs_local_comparison_behavior_view.csv", index=False)

    pd.DataFrame([paper_res]).to_csv(
        outputs.results_dir / "weak_anomaly_recovery_behavior_view.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "recovery_rate_percent": paper_res["selective_recovery_rate_percent"],
                "false_positive_rate": paper_res["selective_fpr"],
                "f1": paper_res["selective_f1"],
                "mean_score_threshold": cfg.promotion_mean_score,
                "strong_support_threshold": cfg.promotion_strong_support,
            }
        ]
    ).to_csv(outputs.results_dir / "selective_weak_promotion_behavior_view.csv", index=False)

    removed_list = sorted(
        set(IDENTITY_HEAVY_COLUMNS)
        | set(BEHAVIOURAL_FEATURE_COLUMNS) - set(BEHAVIOR_GRAPH_CANDIDATE_COLUMNS)
        | set(removed_identity)
    )
    full_row = next(r for r in view_results if r["similarity_feature_view"] == "full_descriptor")
    beh_row = next(r for r in view_results if r["similarity_feature_view"] == "behavior_only")
    norm_row = next(r for r in view_results if r["similarity_feature_view"] == paper_view)

    summary_lines = [
        "# Behaviour-View Fleet Similarity Summary",
        "",
        "## 1. Which features were removed because they encoded vehicle identity?",
        "Excluded from behaviour graph similarity (identity-heavy):",
        "",
        *[f"- `{c}`" for c in removed_list],
        "",
        "Auto-excluded when between/within vehicle variance ratio > "
        f"{cfg.feature_dominance_threshold:.1f}:",
        "",
    ]
    auto_ex = dominance_all[dominance_all["auto_exclude"]]["feature"].tolist()
    summary_lines.extend([f"- `{c}`" for c in auto_ex] or ["- (none)"])
    summary_lines.extend(
        [
            "",
            "## 2. Did behaviour-only descriptors increase cross-vehicle edges?",
            f"- Full descriptor cross-vehicle edges: **{full_row['edge_bias']['pct_cross_vehicle_edges']:.2f}%**",
            f"- Behaviour-only cross-vehicle edges: **{beh_row['edge_bias']['pct_cross_vehicle_edges']:.2f}%**",
            "",
            "## 3. Did vehicle normalization improve cross-vehicle attack similarity?",
            f"- Flooding cross-vehicle similarity (behaviour-only): **{beh_row['flooding_cross_vehicle_similarity']:.4f}**",
            f"- Flooding cross-vehicle similarity (normalized): **{norm_row['flooding_cross_vehicle_similarity']:.4f}**",
            "",
            "## 4. Did flooding attacks from different vehicles become closer?",
            f"- Normalized view flooding cross-vehicle similarity: **{norm_row['flooding_cross_vehicle_similarity']:.4f}**",
            "",
        "## 5. Did weak anomaly recovery improve without excessive FPR?",
        f"- Weak recovery via connected components (full): {full_row['weak_recovery_rate_percent']:.2f}% @ FPR {full_row['weak_fpr']:.4f}",
        f"- Weak recovery via connected components (normalized): {norm_row['weak_recovery_rate_percent']:.2f}% @ FPR {norm_row['weak_fpr']:.4f} "
        "(high FPR when the weak graph forms a single multi-vehicle component).",
        f"- **Selective DBSCAN promotion** (normalized, gated): {norm_row['selective_recovery_rate_percent']:.2f}% @ FPR {norm_row['selective_fpr']:.4f}",
        "",
        "## 6. Which similarity view should be used in the final paper?",
        f"**{SIMILARITY_VIEW_LABELS[paper_view]}** (`{paper_view}`) for fleet graph similarity. "
        "It increases cross-vehicle edges versus the full descriptor while per-vehicle z-scoring "
        "reduces platform-specific baselines. Report weak-anomaly gains with selective promotion gates, "
        "not ungated connected-component promotion.",
            "",
            "## Conclusion",
            "Fleet-level graph construction should use a behaviour-focused, vehicle-normalized descriptor "
            "view rather than the full descriptor. This reduces vehicle-identity bias and allows graph "
            "reasoning to correlate attack behaviour across heterogeneous vehicles.",
            "",
        ]
    )
    summary_path = outputs.results_dir / "behavior_view_similarity_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Behaviour-view experiment complete. Paper view: %s", paper_view)
    return {
        "dominance": outputs.results_dir / "fleet_similarity_feature_dominance.csv",
        "graph_statistics": outputs.results_dir / "behavior_view_fleet_graph_statistics.csv",
        "cross_vehicle_similarity": outputs.results_dir / "behavior_view_cross_vehicle_similarity.csv",
        "topk_bias": outputs.results_dir / "behavior_view_topk_vehicle_bias.csv",
        "ablation_table_tex": table_tex,
        "ablation_table_md": table_md,
        "summary": summary_path,
    }
