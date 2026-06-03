"""Selective weak anomaly promotion with cluster evidence gates and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from src.evaluation.campaign_clustering import extend_dbscan_labels, run_dbscan, subsample_indices
from src.evaluation.weak_anomaly_recovery_experiment import classify_anomaly_strength
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_graph,
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
class SelectivePromotionOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class SelectivePromotionConfig:
    strong_threshold: float = 0.80
    weak_threshold: float = 0.55
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    similarity_threshold: float = 0.95
    min_vehicles: int = 2
    min_cluster_size: int = 5
    min_cohesion: float = 0.97
    mean_score_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80)
    strong_support_thresholds: tuple[float, ...] = (0.40, 0.60, 0.80)
    meaningful_recovery_min_rate: float = 1.0
    max_false_positive_rate: float = 0.50
    max_cohesion_pairs: int = 8000
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 10
    dbscan_pca_components: int = 8
    max_clustering_samples: int | None = 20000
    seed: int = 42
    similarity_feature_view: SimilarityFeatureView = "full_descriptor"
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PromotionGates:
    mean_score_threshold: float
    strong_support_threshold: float
    min_cohesion: float = 0.97
    min_cluster_size: int = 5
    min_vehicles: int = 2


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


def _sampled_mean_pairwise_cohesion(
    X: np.ndarray,
    indices: list[int],
    *,
    max_pairs: int,
    seed: int,
) -> float:
    if len(indices) < 2:
        return 1.0 if len(indices) == 1 else float("nan")
    Xn = normalize(X[indices], norm="l2")
    n = len(indices)
    max_possible = n * (n - 1) // 2
    if max_possible <= max_pairs:
        sims = cosine_similarity(Xn)
        tri = sims[np.triu_indices(n, k=1)]
        return float(np.mean(tri)) if len(tri) else float("nan")
    rng = np.random.default_rng(seed)
    sims: list[float] = []
    for _ in range(max_pairs):
        a, b = rng.integers(0, n, size=2)
        if a == b:
            continue
        sims.append(float(np.dot(Xn[a], Xn[b])))
    return float(np.mean(sims)) if sims else float("nan")


def compute_dbscan_subcluster_statistics(
    descriptors: pd.DataFrame,
    X: np.ndarray,
    labels: np.ndarray,
    *,
    strong_threshold: float,
    max_cohesion_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Per-DBSCAN-cluster statistics (excludes noise label -1)."""
    desc = descriptors.reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    for cid in sorted(int(c) for c in np.unique(labels) if int(c) != -1):
        mask = labels == cid
        sub = desc.loc[mask]
        if sub.empty:
            continue
        row_indices = np.flatnonzero(mask).tolist()
        strong_frac = float((sub["anomaly_score"].astype(float) >= strong_threshold).mean())
        rows.append(
            {
                "cluster_id": cid,
                "cluster_method": "dbscan",
                "cluster_size": int(mask.sum()),
                "number_of_vehicles": int(sub["vehicle_model"].nunique()),
                "mean_anomaly_score": float(sub["anomaly_score"].astype(float).mean()),
                "strong_anomaly_fraction": strong_frac,
                "mean_pairwise_similarity": _sampled_mean_pairwise_cohesion(
                    X,
                    row_indices,
                    max_pairs=max_cohesion_pairs,
                    seed=seed + cid,
                ),
                "num_weak_members": int((sub["anomaly_score"].astype(float) < strong_threshold).sum()),
                "num_strong_members": int((sub["anomaly_score"].astype(float) >= strong_threshold).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_dbscan_on_graph_nodes(
    descriptors: pd.DataFrame,
    X: np.ndarray,
    *,
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_pca_components: int,
    max_clustering_samples: int | None,
    seed: int,
) -> np.ndarray:
    """DBSCAN sub-clusters on cross-vehicle graph node descriptors."""
    meta = descriptors.reset_index(drop=True)
    if max_clustering_samples and len(X) > max_clustering_samples:
        fit_idx = subsample_indices(meta, max_clustering_samples, seed=seed)
        logger.info("DBSCAN fit on %d / %d graph nodes", len(fit_idx), len(X))
    else:
        fit_idx = np.arange(len(X))

    db_fit, projector = run_dbscan(
        X[fit_idx],
        eps=dbscan_eps,
        min_samples=dbscan_min_samples,
        pca_components=dbscan_pca_components,
        random_state=seed,
    )
    return extend_dbscan_labels(X, db_fit, X[fit_idx], projector, eps=dbscan_eps)


def compute_fleet_cluster_statistics(
    G: nx.Graph,
    descriptors: pd.DataFrame,
    X: np.ndarray,
    *,
    strong_threshold: float,
    max_cohesion_pairs: int,
    seed: int,
) -> pd.DataFrame:
    """Connected-component statistics on the full fleet graph."""
    desc = descriptors.reset_index(drop=True)
    eid_to_row = dict(zip(desc["event_id"].astype(str), desc.index.tolist()))
    rows: list[dict[str, Any]] = []

    for cid, component in enumerate(nx.connected_components(G)):
        event_ids = [str(n) for n in component]
        sub = desc[desc["event_id"].astype(str).isin(event_ids)]
        if sub.empty:
            continue
        row_indices = [eid_to_row[eid] for eid in event_ids if eid in eid_to_row]
        strong_frac = float((sub["anomaly_score"].astype(float) >= strong_threshold).mean())
        rows.append(
            {
                "cluster_id": cid,
                "cluster_size": int(len(component)),
                "number_of_vehicles": int(sub["vehicle_model"].nunique()),
                "mean_anomaly_score": float(sub["anomaly_score"].astype(float).mean()),
                "strong_anomaly_fraction": strong_frac,
                "mean_pairwise_similarity": _sampled_mean_pairwise_cohesion(
                    X,
                    row_indices,
                    max_pairs=max_cohesion_pairs,
                    seed=seed + cid,
                ),
                "num_weak_members": int((sub["anomaly_score"].astype(float) < strong_threshold).sum()),
                "num_strong_members": int((sub["anomaly_score"].astype(float) >= strong_threshold).sum()),
            }
        )
    return pd.DataFrame(rows)


def identify_selective_recoverable_weak(
    weak_event_ids: pd.Series,
    weak_cluster_ids: pd.Series,
    clusters: pd.DataFrame,
    gates: PromotionGates,
) -> set[str]:
    """Return weak descriptor IDs eligible for promotion under selective gates."""
    if clusters.empty:
        return set()

    eligible = clusters[
        (clusters["cluster_id"] != -1)
        & (clusters["number_of_vehicles"] >= gates.min_vehicles)
        & (clusters["cluster_size"] >= gates.min_cluster_size)
        & (clusters["mean_anomaly_score"] >= gates.mean_score_threshold)
        & (clusters["mean_pairwise_similarity"] >= gates.min_cohesion)
        & (clusters["strong_anomaly_fraction"] >= gates.strong_support_threshold)
    ]["cluster_id"].astype(int).tolist()

    mask = weak_cluster_ids.isin(eligible) & (weak_cluster_ids != -1)
    return set(weak_event_ids[mask].astype(str).tolist())


def evaluate_selective_promotion(
    weak_df: pd.DataFrame,
    recoverable_ids: set[str],
) -> dict[str, float]:
    """Metrics on the weak-anomaly evaluation subset."""
    y_true = weak_df["ground_truth_label"].astype(int).to_numpy()
    local_pred = weak_df["local_alert"].astype(int).to_numpy()
    fleet_pred = np.where(
        weak_df["descriptor_id"].astype(str).isin(recoverable_ids).to_numpy(), 1, local_pred
    ).astype(int)

    cm = confusion_matrix(y_true, fleet_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    local_cm = confusion_matrix(y_true, local_pred, labels=[0, 1])
    local_fn = int(local_cm[1, 0])

    missed_attack = (local_pred == 0) & (y_true == 1)
    missed_benign = (local_pred == 0) & (y_true == 0)
    in_recoverable = weak_df["descriptor_id"].astype(str).isin(recoverable_ids).to_numpy()
    recovered = missed_attack & in_recoverable
    recovered_benign = missed_benign & in_recoverable
    n_missed_attack = int(missed_attack.sum())
    n_recovered = int(recovered.sum())
    n_recovered_benign = int(recovered_benign.sum())
    recovery_rate = 100.0 * n_recovered / n_missed_attack if n_missed_attack else 0.0

    return {
        "recovery_rate_percent": recovery_rate,
        "recall": float(recall_score(y_true, fleet_pred, zero_division=0)),
        "precision": float(precision_score(y_true, fleet_pred, zero_division=0)),
        "f1": float(f1_score(y_true, fleet_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "recovered_weak_attacks": float(n_recovered),
        "recovered_benign_weak_windows": float(n_recovered_benign),
        "recovery_eligible_weak": float(len(recoverable_ids)),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "local_recall": float(recall_score(y_true, local_pred, zero_division=0)),
        "local_f1": float(f1_score(y_true, local_pred, zero_division=0)),
    }


def _assign_cluster_membership(G: nx.Graph, event_ids: pd.Series) -> pd.Series:
    mapping: dict[str, int] = {}
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            mapping[str(node)] = cid
    return event_ids.astype(str).map(mapping).fillna(-1).astype(int)


def _select_operating_point(
    grid: pd.DataFrame,
    *,
    meaningful_recovery_min_rate: float,
    max_false_positive_rate: float,
) -> tuple[pd.Series, pd.Series | None]:
    """Return (best F1 point, optional precision-safe null-promotion point)."""
    candidates = grid[grid["recovery_rate_percent"] >= meaningful_recovery_min_rate].copy()
    fpr_ok = candidates[candidates["false_positive_rate"] <= max_false_positive_rate]
    if not fpr_ok.empty:
        candidates = fpr_ok
    elif not candidates.empty:
        pass  # no FPR-feasible point; pick best F1 among recovery-feasible
    else:
        candidates = grid[grid["recovery_rate_percent"] > 0].copy()
    if candidates.empty:
        best = grid.sort_values("false_positive_rate").iloc[0]
    else:
        best = candidates.sort_values(
            ["f1", "recovery_rate_percent", "false_positive_rate"],
            ascending=[False, False, True],
        ).iloc[0]

    precision_safe = grid[
        (grid["recovery_rate_percent"] == 0.0) & (grid["false_positive_rate"] == 0.0)
    ]
    ps = precision_safe.sort_values("strong_support_threshold", ascending=False).iloc[0] if len(
        precision_safe
    ) else None
    return best, ps


def run_selective_weak_promotion(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: SelectivePromotionOutputs,
    cfg: SelectivePromotionConfig,
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

    _, G, graph_build_stats, _ = build_cross_vehicle_constrained_graph(
        descriptors,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )
    graph_stats = compute_graph_statistics(G)
    pd.DataFrame([{**graph_stats, **graph_build_stats}]).to_csv(
        outputs.results_dir / "selective_promotion_graph_statistics.csv",
        index=False,
    )

    desc = descriptors.reset_index(drop=True)
    X, _ = resolve_fleet_similarity_matrix(
        desc,
        similarity_feature_view=cfg.similarity_feature_view,
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
        desc,
        X,
        labels,
        strong_threshold=cfg.strong_threshold,
        max_cohesion_pairs=cfg.max_cohesion_pairs,
        seed=cfg.seed,
    )
    cluster_stats.to_csv(outputs.results_dir / "selective_promotion_cluster_statistics.csv", index=False)

    assignments = desc[["event_id", "vehicle_model", "attack_type", "anomaly_score"]].copy()
    assignments["dbscan_cluster_id"] = labels
    assignments.to_csv(outputs.results_dir / "selective_promotion_dbscan_assignments.csv", index=False)

    eid_to_cluster = dict(zip(desc["event_id"].astype(str), labels.astype(int)))
    weak_full = classified[classified["anomaly_strength"] == "weak"].copy()
    weak_df = weak_full[
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
    weak_df = weak_df.rename(columns={"vehicle_model": "vehicle_id", "event_id": "descriptor_id"})
    weak_df["cluster_id"] = weak_df["descriptor_id"].astype(str).map(eid_to_cluster).fillna(-1).astype(int)

    n_dbscan_clusters = int((cluster_stats["number_of_vehicles"] >= cfg.min_vehicles).sum()) if len(
        cluster_stats
    ) else 0
    logger.info(
        "DBSCAN sub-clusters: %d total, %d with >=%d vehicles (noise=%d)",
        len(cluster_stats),
        n_dbscan_clusters,
        cfg.min_vehicles,
        int((labels == -1).sum()),
    )

    grid_rows: list[dict[str, Any]] = []
    for mean_thr, strong_thr in product(cfg.mean_score_thresholds, cfg.strong_support_thresholds):
        gates = PromotionGates(
            mean_score_threshold=float(mean_thr),
            strong_support_threshold=float(strong_thr),
            min_cohesion=cfg.min_cohesion,
            min_cluster_size=cfg.min_cluster_size,
            min_vehicles=cfg.min_vehicles,
        )
        eligible_clusters = cluster_stats[
            (cluster_stats["number_of_vehicles"] >= gates.min_vehicles)
            & (cluster_stats["cluster_size"] >= gates.min_cluster_size)
            & (cluster_stats["mean_anomaly_score"] >= gates.mean_score_threshold)
            & (cluster_stats["mean_pairwise_similarity"] >= gates.min_cohesion)
            & (cluster_stats["strong_anomaly_fraction"] >= gates.strong_support_threshold)
        ]
        recoverable = identify_selective_recoverable_weak(
            weak_df["descriptor_id"],
            weak_df["cluster_id"],
            cluster_stats,
            gates,
        )
        metrics = evaluate_selective_promotion(weak_df, recoverable)
        grid_rows.append(
            {
                "cluster_method": "dbscan",
                "mean_anomaly_score_threshold": mean_thr,
                "strong_support_threshold": strong_thr,
                "strong_support_percent": round(100 * strong_thr, 1),
                "min_cohesion": cfg.min_cohesion,
                "min_cluster_size": cfg.min_cluster_size,
                "min_vehicles": cfg.min_vehicles,
                "eligible_clusters": int(len(eligible_clusters)),
                **metrics,
            }
        )

    sensitivity = pd.DataFrame(grid_rows)
    sensitivity.to_csv(outputs.results_dir / "selective_weak_promotion_sensitivity.csv", index=False)

    best, precision_safe = _select_operating_point(
        sensitivity,
        meaningful_recovery_min_rate=cfg.meaningful_recovery_min_rate,
        max_false_positive_rate=cfg.max_false_positive_rate,
    )
    operating = pd.DataFrame([best.to_dict()])
    operating.to_csv(outputs.results_dir / "selective_weak_promotion_operating_point.csv", index=False)
    if precision_safe is not None:
        pd.DataFrame([precision_safe.to_dict()]).to_csv(
            outputs.results_dir / "selective_weak_promotion_precision_safe.csv",
            index=False,
        )

    best_gates = PromotionGates(
        mean_score_threshold=float(best["mean_anomaly_score_threshold"]),
        strong_support_threshold=float(best["strong_support_threshold"]),
        min_cohesion=cfg.min_cohesion,
        min_cluster_size=cfg.min_cluster_size,
        min_vehicles=cfg.min_vehicles,
    )
    best_recoverable = identify_selective_recoverable_weak(
        weak_df["descriptor_id"],
        weak_df["cluster_id"],
        cluster_stats,
        best_gates,
    )
    promoted = weak_df[weak_df["descriptor_id"].astype(str).isin(best_recoverable)].copy()
    promoted["selectively_promoted"] = 1
    promoted.to_csv(outputs.results_dir / "selective_promoted_weak_anomalies.csv", index=False)

    pivot_f1 = sensitivity.pivot(
        index="mean_anomaly_score_threshold",
        columns="strong_support_percent",
        values="f1",
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Selective Weak Promotion — F1 (DBSCAN sub-clusters)")
    ax.set_xlabel("Strong support (%)")
    ax.set_ylabel("Mean anomaly score threshold")
    _save_figure(fig, outputs.figures_dir / "selective_weak_promotion_sensitivity")

    table_cols = [
        "mean_anomaly_score_threshold",
        "strong_support_percent",
        "recovery_rate_percent",
        "recall",
        "precision",
        "f1",
        "false_positive_rate",
        "eligible_clusters",
        "recovery_eligible_weak",
    ]
    table = sensitivity[table_cols].round(4)
    tex_path = outputs.tables_dir / "table_selective_weak_promotion_sensitivity.tex"
    tex_path.write_text(
        _df_to_ieee_tex(
            table,
            "Sensitivity of selective weak anomaly promotion.",
            "tab:selective-weak-promotion",
        ),
        encoding="utf-8",
    )

    summary_path = outputs.results_dir / "selective_weak_promotion_summary.md"
    summary_path.write_text(
        "\n".join([
            "# Selective Weak Anomaly Promotion — Summary",
            "",
            "## Promotion gates (fixed)",
            f"- **Cluster unit:** DBSCAN sub-clusters on cross-vehicle graph nodes (eps={cfg.dbscan_eps}, min_samples={cfg.dbscan_min_samples})",
            f"- Cluster vehicles ≥ {cfg.min_vehicles}",
            f"- Cluster size ≥ {cfg.min_cluster_size}",
            f"- Mean pairwise similarity (cohesion) ≥ {cfg.min_cohesion}",
            f"- Mean anomaly score threshold (swept): {list(cfg.mean_score_thresholds)}",
            f"- Strong anomaly support (swept): {[f'{int(100*t)}%' for t in cfg.strong_support_thresholds]}",
            "",
            "## Recommended operating point",
            f"- **Mean anomaly score ≥ {best['mean_anomaly_score_threshold']:.2f}**",
            f"- **Strong support ≥ {best['strong_support_percent']:.0f}%**",
            f"- **Recovery rate:** {best['recovery_rate_percent']:.2f}%",
            f"- **Recall:** {best['recall']:.4f} (local baseline {best['local_recall']:.4f})",
            f"- **Precision:** {best['precision']:.4f}",
            f"- **F1:** {best['f1']:.4f} (local baseline {best['local_f1']:.4f})",
            f"- **FPR:** {best['false_positive_rate']:.4f}",
            f"- **Eligible clusters:** {int(best['eligible_clusters']) if pd.notna(best['eligible_clusters']) else 0}",
            f"- **Promoted weak windows:** {int(best['recovery_eligible_weak'])}",
            f"- **Recovered weak attacks:** {int(best['recovered_weak_attacks'])}",
            "",
            "## Interpretation",
            "",
            (
                f"DBSCAN produced **{len(cluster_stats)}** sub-clusters ({int((labels == -1).sum())} noise nodes). "
                "Promotion gates are applied per sub-cluster, not the single connected component."
            ),
            "",
            (
                "The fleet graph forms **one connected component**, but DBSCAN sub-clusters enable selective promotion. "
                + (
                    f"Best F1={best['f1']:.4f} at mean score ≥ {best['mean_anomaly_score_threshold']:.2f}, "
                    f"strong support ≥ {best['strong_support_percent']:.0f}% "
                    f"(recovery {best['recovery_rate_percent']:.2f}%, FPR {best['false_positive_rate']:.4f})."
                    if best["recovery_rate_percent"] > 0
                    else "No configuration achieved meaningful recovery under the swept gates."
                )
            ),
            "",
            (
                f"**Precision-safe alternative:** strong support ≥ {precision_safe['strong_support_percent']:.0f}% "
                f"→ 0 promotions."
                if precision_safe is not None and precision_safe["recovery_rate_percent"] == 0
                else ""
            ),
            "",
            (
                f"No configuration achieves recovery ≥ {cfg.meaningful_recovery_min_rate:.1f}% with FPR ≤ {cfg.max_false_positive_rate:.0%}."
                if best["false_positive_rate"] > cfg.max_false_positive_rate
                and best["recovery_rate_percent"] >= cfg.meaningful_recovery_min_rate
                else ""
            ),
            "",
            "Graph: cross-vehicle constrained kNN (10 same + 5 cross); clustering: DBSCAN on behavioural descriptors.",
            "",
        ]),
        encoding="utf-8",
    )

    logger.info(
        "Selective promotion complete: best F1=%.4f at score>=%.2f, strong>=%.0f%%",
        best["f1"],
        best["mean_anomaly_score_threshold"],
        best["strong_support_percent"],
    )
    return {
        "summary": summary_path,
        "sensitivity": outputs.results_dir / "selective_weak_promotion_sensitivity.csv",
        "operating_point": outputs.results_dir / "selective_weak_promotion_operating_point.csv",
        "sensitivity_figure": outputs.figures_dir / "selective_weak_promotion_sensitivity.png",
        "sensitivity_table": tex_path,
    }
