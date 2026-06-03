"""Compare original top-k vs cross-vehicle constrained kNN fleet graphs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.evaluation.fleet_correlation_experiment import (
    _detection_metrics,
    _pct_gain,
    assign_component_labels,
    evaluate_cluster_quality,
    identify_suspicious_clusters,
)
from src.evaluation.weak_anomaly_recovery_experiment import (
    analyse_weak_clusters,
    classify_anomaly_strength,
    identify_recoverable,
)
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_graph,
    build_fleet_correlation_graph,
    build_networkx_graph,
    build_topk_similarity_edges,
    build_cross_vehicle_constrained_knn_edges,
    compute_graph_statistics,
    load_anomaly_descriptors,
    resolve_fleet_similarity_matrix,
)
from src.graph.fleet_similarity_features import SimilarityFeatureView
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class ComparisonOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class ComparisonConfig:
    top_k_neighbors: int = 15
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    similarity_threshold: float = 0.95
    strong_threshold: float = 0.80
    weak_threshold: float = 0.55
    fleet_minimum_cluster_size: int = 2
    fleet_minimum_vehicle_count: int = 3
    fleet_cluster_score_threshold: float = 0.7
    weak_minimum_cluster_size: int = 2
    weak_minimum_vehicle_count: int = 2
    recovery_score_threshold: float = 0.55
    seed: int = 42
    similarity_feature_view: SimilarityFeatureView = "full_descriptor"
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()


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


def _weak_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def _build_weak_graph_original(
    weak_df: pd.DataFrame,
    *,
    top_k: int,
    similarity_threshold: float,
    seed: int,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> nx.Graph:
    X, _ = resolve_fleet_similarity_matrix(
        weak_df.reset_index(drop=True),
        similarity_feature_view=similarity_feature_view,
        feature_dominance_threshold=feature_dominance_threshold,
        allowed_high_dominance_features=allowed_high_dominance_features,
    )
    _, _, ei, w, sub_idx = build_topk_similarity_edges(
        X, top_k=top_k, similarity_threshold=similarity_threshold, seed=seed
    )
    return build_networkx_graph(weak_df.iloc[sub_idx].reset_index(drop=True), ei, w)


def _build_weak_graph_cross_vehicle(
    weak_df: pd.DataFrame,
    *,
    top_k_same: int,
    top_k_cross: int,
    similarity_threshold: float,
    seed: int,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> nx.Graph:
    sub = weak_df.reset_index(drop=True)
    X, _ = resolve_fleet_similarity_matrix(
        sub,
        similarity_feature_view=similarity_feature_view,
        feature_dominance_threshold=feature_dominance_threshold,
        allowed_high_dominance_features=allowed_high_dominance_features,
    )
    vehicles = sub["vehicle_model"].to_numpy()
    _, _, ei, w, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X,
        vehicles,
        top_k_same_vehicle=top_k_same,
        top_k_cross_vehicle=top_k_cross,
        similarity_threshold=similarity_threshold,
        seed=seed,
    )
    return build_networkx_graph(sub.iloc[sub_idx].reset_index(drop=True), ei, w)


def _evaluate_fleet_correlation(
    G: nx.Graph,
    descriptors: pd.DataFrame,
    cfg: ComparisonConfig,
    cluster_results_path: Path | None,
) -> dict[str, Any]:
    y_true = descriptors["ground_truth_label"].astype(int).to_numpy()
    y_score = descriptors["anomaly_score"].astype(float).to_numpy()
    local_pred = descriptors["local_alert"].astype(int).to_numpy()
    local_m = _detection_metrics(y_true, local_pred, y_score)

    suspicious, _, _ = identify_suspicious_clusters(
        G,
        descriptors,
        minimum_cluster_size=cfg.fleet_minimum_cluster_size,
        minimum_vehicle_count=cfg.fleet_minimum_vehicle_count,
        similarity_threshold=cfg.similarity_threshold,
        fleet_cluster_score_threshold=cfg.fleet_cluster_score_threshold,
        cluster_results_path=cluster_results_path,
    )
    in_cluster = descriptors["event_id"].astype(str).isin(suspicious).to_numpy()
    fleet_pred = ((local_pred == 1) | in_cluster).astype(int)
    fleet_m = _detection_metrics(y_true, fleet_pred, y_score)

    X, _ = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    labels = assign_component_labels(G)
    cq = evaluate_cluster_quality(descriptors, X, labels)
    sil = float(cq.loc[cq["metric"] == "silhouette_score", "value"].iloc[0]) if (
        (cq["metric"] == "silhouette_score").any()
    ) else float("nan")

    return {
        "local_f1": local_m["f1"],
        "fleet_f1": fleet_m["f1"],
        "local_recall": local_m["recall"],
        "fleet_recall": fleet_m["recall"],
        "recall_gain_percent": _pct_gain(fleet_m["recall"], local_m["recall"]),
        "f1_gain_percent": _pct_gain(fleet_m["f1"], local_m["f1"]),
        "fleet_fpr": fleet_m["false_positive_rate"],
        "silhouette_score": sil,
        "promoted_windows": int(((fleet_pred == 1) & (local_pred == 0)).sum()),
    }


def _evaluate_weak_recovery(
    G: nx.Graph,
    weak_table: pd.DataFrame,
    cfg: ComparisonConfig,
) -> dict[str, Any]:
    weak_df, clusters, cross_rate = analyse_weak_clusters(G, weak_table)
    recoverable = identify_recoverable(
        weak_df,
        clusters,
        minimum_cluster_size=cfg.weak_minimum_cluster_size,
        minimum_vehicle_count=cfg.weak_minimum_vehicle_count,
        recovery_score_threshold=cfg.recovery_score_threshold,
    )
    y_true = weak_df["ground_truth_label"].astype(int).to_numpy()
    local_pred = weak_df["local_alert"].astype(int).to_numpy()
    local_m = _weak_metrics(y_true, local_pred)
    fleet_pred = np.where(
        weak_df["descriptor_id"].astype(str).isin(recoverable).to_numpy(), 1, local_pred
    ).astype(int)
    fleet_m = _weak_metrics(y_true, fleet_pred)

    missed_attack = (local_pred == 0) & (y_true == 1)
    n_missed = int(missed_attack.sum())
    recovered = (
        (local_pred == 0)
        & weak_df["descriptor_id"].astype(str).isin(recoverable).to_numpy()
        & (y_true == 1)
    )
    n_recovered = int(recovered.sum())
    recovery_rate = 100.0 * n_recovered / n_missed if n_missed else 0.0

    cross_clusters = int((clusters["number_of_vehicles"] >= 2).sum()) if len(clusters) else 0
    return {
        "weak_recovery_rate_percent": recovery_rate,
        "weak_recall_gain_percent": _pct_gain(fleet_m["recall"], local_m["recall"]),
        "weak_f1_gain_percent": _pct_gain(fleet_m["f1"], local_m["f1"]) if local_m["f1"] > 0 else float("nan"),
        "weak_fleet_fpr": fleet_m["false_positive_rate"],
        "weak_local_fpr": local_m["false_positive_rate"],
        "recovery_eligible": len(recoverable),
        "recovered_attacks": n_recovered,
        "cross_vehicle_weak_clusters": cross_clusters,
        "cross_vehicle_cluster_rate": cross_rate,
    }


def _graph_summary(G: nx.Graph, build_stats: dict[str, float]) -> dict[str, Any]:
    stats = compute_graph_statistics(G)
    return {
        "num_nodes": int(stats["num_nodes"]),
        "num_edges": int(stats["num_edges"]),
        "graph_density": float(stats["graph_density"]),
        "cross_vehicle_edge_count": int(stats["num_cross_vehicle_edges"]),
        "connected_components": int(stats["connected_components"]),
        "cross_vehicle_cluster_count": int(build_stats.get("cross_vehicle_cluster_count", 0)),
    }


def _plot_comparison(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows).set_index("graph_construction")
    metrics = [
        ("cross_vehicle_edge_count", "Cross-vehicle edges"),
        ("cross_vehicle_cluster_count", "Cross-vehicle clusters"),
        ("weak_recovery_rate_percent", "Weak recovery rate (%)"),
        ("recall_gain_percent", "Recall gain (%)"),
        ("f1_gain_percent", "F1 gain (%)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    colors = ["#4472C4", "#ED7D31"]
    x = np.arange(len(df))
    width = 0.35
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].astype(float).to_numpy()
        ax.bar(x, vals, color=colors[: len(df)], width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fleet Graph Construction Comparison", fontsize=11)
    fig.tight_layout()
    _save_figure(fig, out)


def run_graph_construction_comparison(
    *,
    descriptors_path: Path,
    features_path: Path,
    cluster_results_path: Path | None,
    outputs: ComparisonOutputs,
    cfg: ComparisonConfig,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    descriptors = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    classified = classify_anomaly_strength(
        descriptors,
        weak_threshold=cfg.weak_threshold,
        strong_threshold=cfg.strong_threshold,
    )
    weak_df_full = classified.loc[classified["anomaly_strength"] == "weak"].copy()
    weak_table = weak_df_full[
        [
            "window_id",
            "vehicle_model",
            "attack_type",
            "anomaly_score",
            "ground_truth_label",
            "event_id",
            "local_alert",
        ]
    ].copy()
    weak_table = weak_table.rename(columns={"vehicle_model": "vehicle_id", "event_id": "descriptor_id"})

    variants: list[tuple[str, nx.Graph, dict[str, float], str]] = []

    # A) Original top-k (full graph)
    _Gb, G_orig, stats_orig, _ = build_fleet_correlation_graph(
        descriptors,
        top_k_neighbors=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    variants.append(("original_topk", G_orig, stats_orig, "original"))

    # B) Cross-vehicle constrained kNN (full graph)
    _Gb2, G_cv, stats_cv, _ = build_cross_vehicle_constrained_graph(
        descriptors,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    variants.append(("cross_vehicle_knn", G_cv, stats_cv, "cross_vehicle"))

    result_rows: list[dict[str, Any]] = []
    for name, G_full, build_stats, prefix in variants:
        gsum = _graph_summary(G_full, build_stats)
        fleet = _evaluate_fleet_correlation(G_full, descriptors, cfg, cluster_results_path)

        if name == "original_topk":
            G_weak = _build_weak_graph_original(
                weak_df_full,
                top_k=cfg.top_k_neighbors,
                similarity_threshold=cfg.similarity_threshold,
                seed=cfg.seed,
                similarity_feature_view=cfg.similarity_feature_view,
                feature_dominance_threshold=cfg.feature_dominance_threshold,
                allowed_high_dominance_features=cfg.allowed_high_dominance_features,
            )
        else:
            G_weak = _build_weak_graph_cross_vehicle(
                weak_df_full,
                top_k_same=cfg.top_k_same_vehicle,
                top_k_cross=cfg.top_k_cross_vehicle,
                similarity_threshold=cfg.similarity_threshold,
                seed=cfg.seed,
                similarity_feature_view=cfg.similarity_feature_view,
                feature_dominance_threshold=cfg.feature_dominance_threshold,
                allowed_high_dominance_features=cfg.allowed_high_dominance_features,
            )
        weak = _evaluate_weak_recovery(G_weak, weak_table, cfg)

        row = {
            "graph_construction": name,
            **gsum,
            **fleet,
            **weak,
        }
        result_rows.append(row)

        pd.DataFrame([{**gsum, **fleet}]).to_csv(
            outputs.results_dir / f"{prefix}_fleet_graph_statistics.csv",
            index=False,
        )
        pd.DataFrame([fleet]).to_csv(
            outputs.results_dir / f"{prefix}_fleet_correlation_metrics.csv",
            index=False,
        )
        pd.DataFrame([weak]).to_csv(
            outputs.results_dir / f"{prefix}_weak_recovery_metrics.csv",
            index=False,
        )

    comparison = pd.DataFrame(result_rows)
    comparison.to_csv(outputs.results_dir / "graph_construction_comparison.csv", index=False)

    table = comparison[
        [
            "graph_construction",
            "cross_vehicle_edge_count",
            "cross_vehicle_cluster_count",
            "weak_recovery_rate_percent",
            "recall_gain_percent",
            "f1_gain_percent",
            "fleet_f1",
            "fleet_recall",
            "silhouette_score",
        ]
    ].copy()
    table = table.rename(columns={
        "graph_construction": "Graph Construction",
        "cross_vehicle_edge_count": "Cross-Vehicle Edges",
        "cross_vehicle_cluster_count": "Cross-Vehicle Clusters",
        "weak_recovery_rate_percent": "Weak Recovery Rate (\\%)",
        "recall_gain_percent": "Recall Gain (\\%)",
        "f1_gain_percent": "F1 Gain (\\%)",
        "fleet_f1": "Fleet F1",
        "fleet_recall": "Fleet Recall",
        "silhouette_score": "Silhouette",
    })
    tex_path = outputs.tables_dir / "table_graph_construction_comparison.tex"
    tex_path.write_text(
        _df_to_ieee_tex(
            table,
            "Comparison of original top-k vs cross-vehicle constrained kNN fleet graphs.",
            "tab:graph-construction-comparison",
        ),
        encoding="utf-8",
    )

    fig_path = outputs.figures_dir / "cross_vehicle_graph_comparison"
    _plot_comparison(result_rows, fig_path)

    orig = result_rows[0]
    cv = result_rows[1]
    cv_improves_weak = cv["weak_recovery_rate_percent"] > orig["weak_recovery_rate_percent"]
    cv_improves_fleet = cv["recall_gain_percent"] > orig["recall_gain_percent"]
    summary_path = outputs.results_dir / "graph_construction_comparison_summary.md"
    summary_path.write_text(
        "\n".join([
            "# Graph Construction Comparison Summary",
            "",
            "## A) Original Top-K (k=15)",
            f"- Cross-vehicle edges: {orig['cross_vehicle_edge_count']:,}",
            f"- Cross-vehicle clusters (≥2 vehicles): {orig['cross_vehicle_cluster_count']}",
            f"- Weak recovery rate: {orig['weak_recovery_rate_percent']:.2f}%",
            f"- Weak fleet FPR: {orig.get('weak_fleet_fpr', 0):.4f}",
            f"- Fleet recall gain (all descriptors): {orig['recall_gain_percent']:.2f}%",
            f"- Fleet F1 gain: {orig['f1_gain_percent']:.2f}%",
            "",
            "## B) Cross-Vehicle Constrained kNN (10 same + 5 cross)",
            f"- Cross-vehicle edges: {cv['cross_vehicle_edge_count']:,}",
            f"- Cross-vehicle clusters (≥2 vehicles): {cv['cross_vehicle_cluster_count']}",
            f"- Weak recovery rate: {cv['weak_recovery_rate_percent']:.2f}%",
            f"- Weak fleet FPR: {cv.get('weak_fleet_fpr', 0):.4f}",
            f"- Recovered weak attacks: {cv['recovered_attacks']:,} / eligible {cv['recovery_eligible']:,}",
            f"- Fleet recall gain (all descriptors): {cv['recall_gain_percent']:.2f}%",
            f"- Fleet F1 gain: {cv['f1_gain_percent']:.2f}%",
            "",
            "## Conclusion",
            "",
            (
                "**Cross-vehicle constrained kNN fixes the connectivity problem** "
                f"(cross-vehicle edges {orig['cross_vehicle_edge_count']:,} → {cv['cross_vehicle_edge_count']:,}) "
                "and enables weak-anomaly recovery when cluster gates are applied on the weak-only graph."
                if cv_improves_weak
                else "**Cross-vehicle kNN increases cross-vehicle edges** but does not improve recovery under current gates."
            ),
            "",
            (
                f"On the **full descriptor graph**, fleet correlation gain remains **{cv['recall_gain_percent']:.2f}%** "
                f"because promotion thresholds (≥{cfg.fleet_minimum_vehicle_count} vehicles, mean score ≥{cfg.fleet_cluster_score_threshold}) "
                "are not satisfied—or the graph collapses to one component whose mean score is below threshold."
                if not cv_improves_fleet
                else f"Full-graph fleet recall improved by {cv['recall_gain_percent']:.2f}%."
            ),
            "",
            (
                f"**FPR trade-off on weak anomalies:** cross-vehicle kNN recovery raises weak-subset FPR "
                f"from {orig.get('weak_local_fpr', 0):.4f} to {cv.get('weak_fleet_fpr', 0):.4f} "
                "when the weak graph forms a single multi-vehicle component—report both recovery and FPR in the paper."
                if cv.get("weak_fleet_fpr", 0) > orig.get("weak_local_fpr", 0) + 0.01
                else ""
            ),
            "",
        ]),
        encoding="utf-8",
    )

    logger.info("Graph construction comparison complete.")
    return {
        "comparison_table": tex_path,
        "comparison_figure": fig_path.with_suffix(".png"),
        "summary": summary_path,
    }
