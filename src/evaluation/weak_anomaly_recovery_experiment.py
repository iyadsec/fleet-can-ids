"""Weak anomaly recovery evaluation (local IDS vs fleet behavioural correlation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.graph.fleet_graph_builder import (
    build_networkx_graph,
    build_topk_similarity_edges,
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


@dataclass(frozen=True)
class WeakRecoveryOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class WeakRecoveryConfig:
    strong_threshold: float = 0.80
    weak_threshold: float = 0.55
    top_k_neighbors: int = 15
    similarity_threshold: float = 0.95
    minimum_cluster_size: int = 2
    minimum_vehicle_count: int = 2
    recovery_score_threshold: float = 0.55
    max_viz_nodes: int = 1200
    seed: int = 42
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


def _pct_gain(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("nan")
    return 100.0 * (new - old) / old


def _weak_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def classify_anomaly_strength(
    df: pd.DataFrame,
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> pd.DataFrame:
    """Label weak vs strong anomalies from configured score bands."""
    out = df.copy()
    score = out["anomaly_score"].astype(float)
    out["anomaly_strength"] = np.where(
        score >= strong_threshold,
        "strong",
        np.where(score >= weak_threshold, "weak", "below_weak"),
    )
    return out


def build_weak_anomaly_graph(
    weak_df: pd.DataFrame,
    *,
    top_k: int,
    similarity_threshold: float,
    seed: int,
    similarity_feature_view: SimilarityFeatureView = "full_descriptor",
    feature_dominance_threshold: float = 5.0,
    allowed_high_dominance_features: frozenset[str] = frozenset(),
) -> tuple[nx.Graph, dict[str, Any]]:
    """Top-k cosine graph on weak-anomaly descriptors only."""
    X, _ = resolve_fleet_similarity_matrix(
        weak_df.reset_index(drop=True),
        similarity_feature_view=similarity_feature_view,
        feature_dominance_threshold=feature_dominance_threshold,
        allowed_high_dominance_features=allowed_high_dominance_features,
    )
    _, _, ei_after, w_after, sub_idx = build_topk_similarity_edges(
        X,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metric="cosine",
        max_nodes=None,
        seed=seed,
    )
    df_sub = weak_df.iloc[sub_idx].reset_index(drop=True)
    G = build_networkx_graph(df_sub, ei_after, w_after)

    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components] if components else [0]
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = float(nx.density(G)) if n > 1 else 0.0

    stats = {
        "node_count": int(n),
        "edge_count": int(m),
        "graph_density": density,
        "connected_components": int(len(components)),
        "average_cluster_size": float(np.mean(comp_sizes)),
        "max_cluster_size": int(max(comp_sizes, default=0)),
        "top_k_neighbors": int(top_k),
        "similarity_threshold": float(similarity_threshold),
    }
    return G, stats


def analyse_weak_clusters(
    G: nx.Graph,
    weak_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Summarise connected components and attach cluster_id to each weak descriptor."""
    weak_df = weak_df.copy()
    event_to_cluster: dict[str, int] = {}
    cluster_rows: list[dict[str, Any]] = []

    for cid, component in enumerate(nx.connected_components(G)):
        event_ids = list(component)
        for eid in event_ids:
            event_to_cluster[str(eid)] = cid
        sub = weak_df[weak_df["descriptor_id"].astype(str).isin(event_ids)]
        if sub.empty:
            continue
        attacks = sorted(sub["attack_type"].dropna().unique().tolist())
        cluster_rows.append(
            {
                "cluster_id": cid,
                "cluster_size": len(component),
                "number_of_vehicles": int(sub["vehicle_id"].nunique()),
                "mean_anomaly_score": float(sub["anomaly_score"].mean()),
                "attack_labels_present": ",".join(attacks),
                "num_attack_windows": int(sub["ground_truth_label"].astype(int).sum()),
            }
        )

    clusters = pd.DataFrame(cluster_rows)
    weak_df["cluster_id"] = weak_df["descriptor_id"].astype(str).map(event_to_cluster)
    weak_df["cluster_id"] = weak_df["cluster_id"].fillna(-1).astype(int)

    if len(clusters):
        cross_rate = float((clusters["number_of_vehicles"] >= 2).mean())
    else:
        cross_rate = 0.0

    return weak_df, clusters, cross_rate


def identify_recoverable(
    weak_df: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    minimum_cluster_size: int,
    minimum_vehicle_count: int,
    recovery_score_threshold: float,
) -> set[str]:
    """Weak anomalies eligible for fleet recovery (evidence gate — not yet promoted)."""
    if clusters.empty:
        return set()

    eligible_clusters = clusters[
        (clusters["number_of_vehicles"] >= minimum_vehicle_count)
        & (clusters["cluster_size"] >= minimum_cluster_size)
        & (clusters["mean_anomaly_score"] >= recovery_score_threshold)
    ]["cluster_id"].astype(int).tolist()

    recoverable = weak_df[weak_df["cluster_id"].isin(eligible_clusters)]
    return set(recoverable["descriptor_id"].astype(str).tolist())


def _plot_weak_clusters(
    G: nx.Graph,
    weak_df: pd.DataFrame,
    recovered_ids: set[str],
    recoverable_ids: set[str],
    out: Path,
    *,
    max_nodes: int,
    seed: int,
) -> None:
    """Visualise weak-anomaly graph; highlight cross-vehicle clusters and recovery."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        logger.warning("Weak graph too small for visualisation")
        return

    rng = np.random.default_rng(seed)
    if len(nodes) > max_nodes:
        # Prefer nodes in multi-vehicle clusters and recoverable nodes
        priority = weak_df.set_index("descriptor_id")
        scored: list[tuple[str, int]] = []
        for n in nodes:
            score = 0
            if str(n) in recovered_ids:
                score += 3
            elif str(n) in recoverable_ids:
                score += 2
            if str(n) in priority.index:
                cid = int(priority.loc[str(n), "cluster_id"])
                cv = weak_df[weak_df["cluster_id"] == cid]["vehicle_id"].nunique()
                if cv >= 2:
                    score += 1
            scored.append((str(n), score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        keep = {n for n, _ in scored[:max_nodes]}
        for n in list(keep):
            keep.update(G.neighbors(n))
        keep = set(list(keep)[:max_nodes])
        H = G.subgraph(keep).copy()
    else:
        H = G

    pos = nx.spring_layout(H, seed=seed, k=0.25, iterations=60)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    red = [n for n in H.nodes if str(n) in recovered_ids]
    blue = [n for n in H.nodes if str(n) not in recovered_ids]

    if blue:
        nx.draw_networkx_nodes(H, pos, nodelist=blue, node_color="#4472C4", node_size=22, alpha=0.75, ax=ax)
    if red:
        nx.draw_networkx_nodes(H, pos, nodelist=red, node_color="#C00000", node_size=28, alpha=0.9, ax=ax)

    # Highlight edges within multi-vehicle clusters
    wdf = weak_df.set_index("descriptor_id")
    cross_edges = []
    for u, v in H.edges():
        if str(u) not in wdf.index or str(v) not in wdf.index:
            continue
        if wdf.loc[str(u), "cluster_id"] == wdf.loc[str(v), "cluster_id"]:
            cid = int(wdf.loc[str(u), "cluster_id"])
            if weak_df[weak_df["cluster_id"] == cid]["vehicle_id"].nunique() >= 2:
                cross_edges.append((u, v))
    if cross_edges:
        nx.draw_networkx_edges(H, pos, edgelist=cross_edges, width=0.6, alpha=0.35, edge_color="#333333", ax=ax)
    other_edges = [(u, v) for u, v in H.edges() if (u, v) not in cross_edges and (v, u) not in cross_edges]
    if other_edges:
        nx.draw_networkx_edges(H, pos, edgelist=other_edges, width=0.2, alpha=0.08, ax=ax)

    ax.scatter([], [], c="#C00000", s=40, label="Recovered weak anomalies")
    ax.scatter([], [], c="#4472C4", s=40, label="Non-recovered weak anomalies")
    ax.set_title(
        f"Weak Anomaly Clusters (n={H.number_of_nodes()}, m={H.number_of_edges()})",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.axis("off")
    _save_figure(fig, out)


def run_weak_anomaly_recovery_experiment(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: WeakRecoveryOutputs,
    cfg: WeakRecoveryConfig,
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

    weak_mask = classified["anomaly_strength"] == "weak"
    strong_mask = classified["anomaly_strength"] == "strong"
    n_weak = int(weak_mask.sum())
    n_strong = int(strong_mask.sum())

    weak_table = classified.loc[weak_mask, [
        "window_id", "vehicle_model", "anomaly_score", "ground_truth_label", "event_id", "attack_type", "local_alert",
    ]].copy()
    weak_table = weak_table.rename(columns={"vehicle_model": "vehicle_id", "event_id": "descriptor_id"})
    weak_table.to_csv(outputs.results_dir / "weak_anomalies.csv", index=False)

    counts = pd.DataFrame([
        {"category": "weak_anomalies", "count": n_weak},
        {"category": "strong_anomalies", "count": n_strong},
    ])
    counts.to_csv(outputs.results_dir / "weak_strong_anomaly_counts.csv", index=False)

    weak_only = classified.loc[weak_mask].copy()
    G, graph_stats = build_weak_anomaly_graph(
        weak_only,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    pd.DataFrame([graph_stats]).to_csv(
        outputs.results_dir / "weak_anomaly_graph_statistics.csv",
        index=False,
    )

    weak_df, clusters, cross_vehicle_rate = analyse_weak_clusters(G, weak_table)
    clusters.to_csv(outputs.results_dir / "weak_anomaly_clusters.csv", index=False)

    cluster_vehicle_summary = pd.DataFrame([
        {
            "vehicle_bucket": "1_vehicle",
            "cluster_count": int((clusters["number_of_vehicles"] == 1).sum()) if len(clusters) else 0,
        },
        {
            "vehicle_bucket": "2_vehicles",
            "cluster_count": int((clusters["number_of_vehicles"] == 2).sum()) if len(clusters) else 0,
        },
        {
            "vehicle_bucket": "3_plus_vehicles",
            "cluster_count": int((clusters["number_of_vehicles"] >= 3).sum()) if len(clusters) else 0,
        },
        {"vehicle_bucket": "cross_vehicle_cluster_rate", "cluster_count": round(cross_vehicle_rate, 6)},
    ])
    cluster_vehicle_summary.to_csv(
        outputs.results_dir / "weak_anomaly_cluster_vehicle_summary.csv",
        index=False,
    )

    recoverable_ids = identify_recoverable(
        weak_df,
        clusters,
        minimum_cluster_size=cfg.minimum_cluster_size,
        minimum_vehicle_count=cfg.minimum_vehicle_count,
        recovery_score_threshold=cfg.recovery_score_threshold,
    )
    recoverable_df = weak_df[weak_df["descriptor_id"].astype(str).isin(recoverable_ids)].copy()
    recoverable_df["recovery_eligible"] = 1
    recoverable_df.to_csv(outputs.results_dir / "recoverable_weak_anomalies.csv", index=False)

    y_true = weak_df["ground_truth_label"].astype(int).to_numpy()
    local_pred = weak_df["local_alert"].astype(int).to_numpy()
    local_m = _weak_metrics(y_true, local_pred)
    pd.DataFrame([{"system": "local_only", **local_m}]).to_csv(
        outputs.results_dir / "weak_anomaly_local_metrics.csv",
        index=False,
    )

    fleet_pred = local_pred.copy()
    recoverable_idx = weak_df["descriptor_id"].astype(str).isin(recoverable_ids).to_numpy()
    fleet_pred = np.where(recoverable_idx, 1, fleet_pred).astype(int)
    fleet_m = _weak_metrics(y_true, fleet_pred)
    pd.DataFrame([{"system": "fleet_aware", **fleet_m}]).to_csv(
        outputs.results_dir / "weak_anomaly_fleet_metrics.csv",
        index=False,
    )

    missed_locally = weak_df["local_alert"].astype(int) == 0
    missed_attack = missed_locally & (weak_df["ground_truth_label"].astype(int) == 1)
    n_missed_attack = int(missed_attack.sum())

    newly_recovered = (
        missed_locally
        & weak_df["descriptor_id"].astype(str).isin(recoverable_ids)
        & (weak_df["ground_truth_label"].astype(int) == 1)
    )
    n_newly_recovered = int(newly_recovered.sum())
    recovered_ids = set(weak_df.loc[newly_recovered, "descriptor_id"].astype(str).tolist())

    recovery_rate = (100.0 * n_newly_recovered / n_missed_attack) if n_missed_attack else 0.0
    recovery_df = pd.DataFrame([
        {
            "metric": "weak_anomalies_total",
            "value": n_weak,
        },
        {
            "metric": "weak_anomalies_missed_by_local_ids",
            "value": int(missed_locally.sum()),
        },
        {
            "metric": "weak_attack_anomalies_missed_by_local_ids",
            "value": n_missed_attack,
        },
        {
            "metric": "recovery_eligible_weak_anomalies",
            "value": len(recoverable_ids),
        },
        {
            "metric": "newly_recovered_weak_attack_anomalies",
            "value": n_newly_recovered,
        },
        {
            "metric": "weak_anomaly_recovery_rate_percent",
            "value": round(recovery_rate, 4),
        },
        {
            "metric": "recall_gain_percent",
            "value": round(_pct_gain(fleet_m["recall"], local_m["recall"]), 4),
        },
        {
            "metric": "f1_gain_percent",
            "value": round(_pct_gain(fleet_m["f1"], local_m["f1"]), 4),
        },
    ])
    recovery_df.to_csv(outputs.results_dir / "weak_anomaly_recovery.csv", index=False)

    comparison = pd.DataFrame([
        {
            "Metric": "Recall",
            "Local IDS": round(local_m["recall"], 4),
            "Fleet-Aware IDS": round(fleet_m["recall"], 4),
            "Improvement (%)": round(_pct_gain(fleet_m["recall"], local_m["recall"]), 2),
        },
        {
            "Metric": "Precision",
            "Local IDS": round(local_m["precision"], 4),
            "Fleet-Aware IDS": round(fleet_m["precision"], 4),
            "Improvement (%)": round(_pct_gain(fleet_m["precision"], local_m["precision"]), 2),
        },
        {
            "Metric": "F1-score",
            "Local IDS": round(local_m["f1"], 4),
            "Fleet-Aware IDS": round(fleet_m["f1"], 4),
            "Improvement (%)": round(_pct_gain(fleet_m["f1"], local_m["f1"]), 2),
        },
        {
            "Metric": "False Positive Rate",
            "Local IDS": round(local_m["false_positive_rate"], 4),
            "Fleet-Aware IDS": round(fleet_m["false_positive_rate"], 4),
            "Improvement (%)": round(_pct_gain(fleet_m["false_positive_rate"], local_m["false_positive_rate"]), 2),
        },
        {
            "Metric": "Weak Anomaly Recovery Rate",
            "Local IDS": 0.0,
            "Fleet-Aware IDS": round(recovery_rate, 4),
            "Improvement (%)": round(recovery_rate, 2),
        },
    ])
    caption = "Weak anomaly detection: local-only vs fleet-aware recovery."
    (outputs.tables_dir / "table_weak_anomaly_recovery.tex").write_text(
        _df_to_ieee_tex(comparison, caption, "tab:weak-anomaly-recovery"),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_weak_anomaly_recovery.md").write_text(
        "# Weak Anomaly Detection: Local-Only vs Fleet-Aware\n\n" + _df_to_markdown(comparison),
        encoding="utf-8",
    )

    fig_path = outputs.figures_dir / "weak_anomaly_cluster_visualisation"
    _plot_weak_clusters(
        G,
        weak_df,
        recovered_ids,
        recoverable_ids,
        fig_path,
        max_nodes=cfg.max_viz_nodes,
        seed=cfg.seed,
    )

    cross_vehicle_clusters = int((clusters["number_of_vehicles"] >= 2).sum()) if len(clusters) else 0
    fpr_delta = fleet_m["false_positive_rate"] - local_m["false_positive_rate"]
    supports_hypothesis = (
        cross_vehicle_clusters > 0
        and len(recoverable_ids) > 0
        and n_newly_recovered > 0
        and fleet_m["recall"] > local_m["recall"]
    )

    if supports_hypothesis:
        hypothesis_verdict = (
            "**The experimental evidence supports the hypothesis:** weak anomalies missed locally "
            f"can be recovered through fleet-level behavioural correlation "
            f"({n_newly_recovered} attack windows recovered, recovery rate {recovery_rate:.2f}%)."
        )
    elif cross_vehicle_clusters > 0 and len(recoverable_ids) > 0:
        hypothesis_verdict = (
            "**Cross-vehicle weak-anomaly clusters exist**, but recovery under the configured gates "
            f"did not yield meaningful attack recovery ({n_newly_recovered} attack windows recovered). "
            "The hypothesis is not fully supported on this dataset."
        )
    elif cross_vehicle_clusters > 0:
        hypothesis_verdict = (
            "**Weak anomalies form cross-vehicle behavioural clusters**, but none met recovery "
            "eligibility thresholds — fleet promotion was not applied."
        )
    else:
        hypothesis_verdict = (
            "**The experimental evidence does not support the hypothesis** on this dataset: "
            "weak anomalies did not form meaningful cross-vehicle clusters under the configured graph settings."
        )

    summary_path = outputs.results_dir / "weak_anomaly_recovery_summary.md"
    summary_path.write_text(
        "\n".join([
            "# Weak Anomaly Recovery — Summary",
            "",
            "## Anomaly counts",
            f"- **Total weak anomalies** ({cfg.weak_threshold} ≤ score < {cfg.strong_threshold}): {n_weak:,}",
            f"- **Total strong anomalies** (score ≥ {cfg.strong_threshold}): {n_strong:,}",
            "",
            "## Cross-vehicle cluster evidence (Step 2–3)",
            f"- **Weak-anomaly graph nodes:** {graph_stats['node_count']:,}",
            f"- **Edges (top-k={cfg.top_k_neighbors}, sim ≥ {cfg.similarity_threshold}):** {graph_stats['edge_count']:,}",
            f"- **Graph density:** {graph_stats['graph_density']:.8f}",
            f"- **Connected components:** {graph_stats['connected_components']:,}",
            f"- **Average cluster size:** {graph_stats['average_cluster_size']:.2f}",
            f"- **Cross-vehicle cluster rate:** {cross_vehicle_rate:.2%}",
            f"- **Clusters with 1 / 2 / 3+ vehicles:** "
            f"{int((clusters['number_of_vehicles'] == 1).sum()) if len(clusters) else 0} / "
            f"{int((clusters['number_of_vehicles'] == 2).sum()) if len(clusters) else 0} / "
            f"{int((clusters['number_of_vehicles'] >= 3).sum()) if len(clusters) else 0}",
            "",
            "## Recovery (Step 4–7)",
            f"- **Recovery eligible** (≥{cfg.minimum_vehicle_count} vehicles, size ≥{cfg.minimum_cluster_size}, "
            f"mean score ≥{cfg.recovery_score_threshold}): {len(recoverable_ids):,}",
            f"- **Actually recovered attack weak anomalies:** {n_newly_recovered:,}",
            f"- **Weak attack anomalies missed locally:** {n_missed_attack:,}",
            f"- **Recovery rate:** {recovery_rate:.2f}%",
            f"- **Recall gain:** {_pct_gain(fleet_m['recall'], local_m['recall']):.2f}%",
            f"- **F1 gain:** {_pct_gain(fleet_m['f1'], local_m['f1']):.2f}%",
            "",
            "## False-positive impact",
            f"- **Local FPR (weak subset):** {local_m['false_positive_rate']:.4f}",
            f"- **Fleet FPR (weak subset):** {fleet_m['false_positive_rate']:.4f}",
            f"- **FPR change:** {fpr_delta:+.4f}",
            "",
            "## Conclusion",
            "",
            "### Do weak anomalies form cross-vehicle clusters?",
            (
                f"Yes — {cross_vehicle_clusters} connected components span ≥2 vehicles "
                f"(cross-vehicle rate {cross_vehicle_rate:.2%})."
                if cross_vehicle_clusters > 0
                else "No meaningful cross-vehicle clusters were observed under the current graph construction."
            ),
            "",
            hypothesis_verdict,
            "",
            "Parameters: "
            f"weak_threshold={cfg.weak_threshold}, strong_threshold={cfg.strong_threshold}, "
            f"top_k={cfg.top_k_neighbors}, similarity_threshold={cfg.similarity_threshold}, "
            f"minimum_vehicle_count={cfg.minimum_vehicle_count}, "
            f"recovery_score_threshold={cfg.recovery_score_threshold}.",
            "",
        ]),
        encoding="utf-8",
    )

    logger.info(
        "Weak recovery experiment complete: %d weak, %d recoverable, %d recovered attacks",
        n_weak,
        len(recoverable_ids),
        n_newly_recovered,
    )
    return {
        "weak_anomaly_recovery_summary": summary_path,
        "table_weak_anomaly_recovery": outputs.tables_dir / "table_weak_anomaly_recovery.tex",
        "weak_anomaly_cluster_visualisation": fig_path.with_suffix(".png"),
    }
