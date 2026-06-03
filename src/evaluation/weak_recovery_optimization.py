"""Full grid search over DBSCAN + promotion parameters for weak anomaly recovery."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.graph_construction_comparison import (
    _build_weak_graph_cross_vehicle,
    _build_weak_graph_original,
)
from src.evaluation.selective_weak_promotion import (
    compute_dbscan_subcluster_statistics,
    evaluate_selective_promotion,
    run_dbscan_on_graph_nodes,
)
from src.evaluation.weak_anomaly_recovery_experiment import (
    _df_to_ieee_tex,
    analyse_weak_clusters,
    classify_anomaly_strength,
    identify_recoverable,
)
from src.graph.fleet_graph_builder import (
    build_cross_vehicle_constrained_graph,
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

DEFAULT_DBSCAN_EPS_VALUES: tuple[float, ...] = (
    0.03,
    0.05,
    0.07,
    0.10,
    0.12,
    0.15,
    0.20,
    0.25,
    0.35,
    0.50,
    0.70,
    0.90,
    1.0,
    1.2,
    1.5,
    2.0,
)
DEFAULT_DBSCAN_MIN_SAMPLES_VALUES: tuple[int, ...] = (3, 5, 10, 15)
DEFAULT_MINIMUM_VEHICLE_COUNT_VALUES: tuple[int, ...] = (2, 3)
DEFAULT_MINIMUM_CLUSTER_SIZE_VALUES: tuple[int, ...] = (3, 5, 10)
DEFAULT_CLUSTER_MEAN_SCORE_VALUES: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DEFAULT_CLUSTER_COHESION_VALUES: tuple[float, ...] = (0.90, 0.93, 0.95, 0.97)
DEFAULT_STRONG_SUPPORT_VALUES: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


@dataclass(frozen=True)
class WeakRecoveryOptimizationOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class WeakRecoveryOptimizationConfig:
    strong_threshold: float = 0.80
    weak_threshold: float = 0.55
    top_k_same_vehicle: int = 10
    top_k_cross_vehicle: int = 5
    top_k_neighbors: int = 15
    similarity_threshold: float = 0.95
    weak_minimum_cluster_size: int = 2
    weak_minimum_vehicle_count: int = 2
    recovery_score_threshold: float = 0.55
    dbscan_pca_components: int = 8
    max_clustering_samples: int | None = 20000
    max_cohesion_pairs: int = 8000
    ieee_recovery_min_percent: float = 10.0
    balanced_max_fpr: float = 0.10
    conservative_max_fpr: float = 0.05
    dbscan_eps_values: tuple[float, ...] = DEFAULT_DBSCAN_EPS_VALUES
    dbscan_min_samples_values: tuple[int, ...] = DEFAULT_DBSCAN_MIN_SAMPLES_VALUES
    minimum_vehicle_count_values: tuple[int, ...] = DEFAULT_MINIMUM_VEHICLE_COUNT_VALUES
    minimum_cluster_size_values: tuple[int, ...] = DEFAULT_MINIMUM_CLUSTER_SIZE_VALUES
    cluster_mean_score_threshold_values: tuple[float, ...] = DEFAULT_CLUSTER_MEAN_SCORE_VALUES
    cluster_cohesion_threshold_values: tuple[float, ...] = DEFAULT_CLUSTER_COHESION_VALUES
    strong_support_threshold_values: tuple[float, ...] = DEFAULT_STRONG_SUPPORT_VALUES
    seed: int = 42
    similarity_feature_view: SimilarityFeatureView = "full_descriptor"
    feature_dominance_threshold: float = 5.0
    allowed_high_dominance_features: frozenset[str] = frozenset()


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _count_cross_vehicle_clusters(
    cluster_stats: pd.DataFrame,
    minimum_vehicle_count: int,
) -> int:
    if cluster_stats.empty:
        return 0
    return int((cluster_stats["number_of_vehicles"] >= minimum_vehicle_count).sum())


def _metrics_from_promoted_mask(
    *,
    y_true: np.ndarray,
    local_pred: np.ndarray,
    promoted: np.ndarray,
    missed_attack: np.ndarray,
    missed_benign: np.ndarray,
    n_missed_attack: int,
) -> dict[str, float]:
    """Compute promotion metrics from a boolean promoted mask (vectorized)."""
    fleet_pred = np.where(promoted, 1, local_pred).astype(int)
    tp = int(((fleet_pred == 1) & (y_true == 1)).sum())
    fp = int(((fleet_pred == 1) & (y_true == 0)).sum())
    fn = int(((fleet_pred == 0) & (y_true == 1)).sum())
    tn = int(((fleet_pred == 0) & (y_true == 0)).sum())
    n_recovered = int((missed_attack & promoted).sum())
    recovery_rate = 100.0 * n_recovered / n_missed_attack if n_missed_attack else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    n_promoted = int(promoted.sum())
    return {
        "recovery_rate_percent": recovery_rate,
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "recovered_weak_attacks": float(n_recovered),
        "recovered_benign_weak_windows": float((missed_benign & promoted).sum()),
        "recovery_eligible_weak": float(n_promoted),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _sweep_promotion_params(
    *,
    dbscan_eps: float,
    dbscan_min_samples: int,
    cluster_stats: pd.DataFrame,
    weak_df: pd.DataFrame,
    min_vehicle_values: tuple[int, ...],
    min_cluster_values: tuple[int, ...],
    mean_score_values: tuple[float, ...],
    cohesion_values: tuple[float, ...],
    strong_support_values: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Fast sweep of promotion gates for a fixed DBSCAN clustering."""
    rows: list[dict[str, Any]] = []
    cross_vehicle_by_min_vehicles = {
        mv: _count_cross_vehicle_clusters(cluster_stats, mv) for mv in min_vehicle_values
    }

    y_true = weak_df["ground_truth_label"].astype(int).to_numpy()
    local_pred = weak_df["local_alert"].astype(int).to_numpy()
    weak_clusters = weak_df["cluster_id"].to_numpy()
    missed_attack = (local_pred == 0) & (y_true == 1)
    missed_benign = (local_pred == 0) & (y_true == 0)
    n_missed_attack = int(missed_attack.sum())
    local_tp = int(((local_pred == 1) & (y_true == 1)).sum())
    local_fp = int(((local_pred == 1) & (y_true == 0)).sum())
    local_fn = int(((local_pred == 0) & (y_true == 1)).sum())
    local_recall = local_tp / (local_tp + local_fn) if (local_tp + local_fn) else 0.0
    local_prec = local_tp / (local_tp + local_fp) if (local_tp + local_fp) else 0.0
    local_f1 = (
        2 * local_prec * local_recall / (local_prec + local_recall)
        if (local_prec + local_recall)
        else 0.0
    )

    if cluster_stats.empty:
        zero_metrics = _metrics_from_promoted_mask(
            y_true=y_true,
            local_pred=local_pred,
            promoted=np.zeros(len(y_true), dtype=bool),
            missed_attack=missed_attack,
            missed_benign=missed_benign,
            n_missed_attack=n_missed_attack,
        )
        zero_metrics["local_recall"] = local_recall
        zero_metrics["local_f1"] = float(local_f1)
        for (
            min_vehicles,
            min_cluster_size,
            mean_thr,
            cohesion_thr,
            strong_thr,
        ) in product(
            min_vehicle_values,
            min_cluster_values,
            mean_score_values,
            cohesion_values,
            strong_support_values,
        ):
            rows.append(
                {
                    "dbscan_eps": dbscan_eps,
                    "dbscan_min_samples": dbscan_min_samples,
                    "minimum_vehicle_count": min_vehicles,
                    "minimum_cluster_size": min_cluster_size,
                    "cluster_mean_score_threshold": mean_thr,
                    "cluster_cohesion_threshold": cohesion_thr,
                    "strong_support_threshold": strong_thr,
                    "eligible_cluster_count": 0,
                    "cross_vehicle_cluster_count": cross_vehicle_by_min_vehicles[min_vehicles],
                    **zero_metrics,
                }
            )
        return rows

    cids = cluster_stats["cluster_id"].astype(int).to_numpy()
    n_veh = cluster_stats["number_of_vehicles"].to_numpy()
    csize = cluster_stats["cluster_size"].to_numpy()
    mean_sc = cluster_stats["mean_anomaly_score"].to_numpy()
    cohesion_arr = cluster_stats["mean_pairwise_similarity"].to_numpy()
    strong_frac = cluster_stats["strong_anomaly_fraction"].to_numpy()

    for (
        min_vehicles,
        min_cluster_size,
        mean_thr,
        cohesion_thr,
        strong_thr,
    ) in product(
        min_vehicle_values,
        min_cluster_values,
        mean_score_values,
        cohesion_values,
        strong_support_values,
    ):
        gate_mask = (
            (n_veh >= min_vehicles)
            & (csize >= min_cluster_size)
            & (mean_sc >= mean_thr)
            & (cohesion_arr >= cohesion_thr)
            & (strong_frac >= strong_thr)
        )
        eligible_cids = cids[gate_mask]
        promoted = np.isin(weak_clusters, eligible_cids) & (weak_clusters != -1)
        metrics = _metrics_from_promoted_mask(
            y_true=y_true,
            local_pred=local_pred,
            promoted=promoted,
            missed_attack=missed_attack,
            missed_benign=missed_benign,
            n_missed_attack=n_missed_attack,
        )
        metrics["local_recall"] = local_recall
        metrics["local_f1"] = float(local_f1)
        rows.append(
            {
                "dbscan_eps": dbscan_eps,
                "dbscan_min_samples": dbscan_min_samples,
                "minimum_vehicle_count": min_vehicles,
                "minimum_cluster_size": min_cluster_size,
                "cluster_mean_score_threshold": mean_thr,
                "cluster_cohesion_threshold": cohesion_thr,
                "strong_support_threshold": strong_thr,
                "eligible_cluster_count": int(gate_mask.sum()),
                "cross_vehicle_cluster_count": cross_vehicle_by_min_vehicles[min_vehicles],
                **metrics,
            }
        )
    return rows


def _select_operating_points(
    sweep: pd.DataFrame,
    *,
    ieee_recovery_min_percent: float,
    balanced_max_fpr: float,
    conservative_max_fpr: float,
) -> dict[str, pd.Series]:
    """Pick Maximum Recovery, Balanced, Conservative, and IEEE Recommended configs."""
    sort_cols = ["f1", "recovery_rate_percent", "false_positive_rate"]

    max_recovery = sweep.sort_values(
        ["recovery_rate_percent", "f1", "false_positive_rate"],
        ascending=[False, False, True],
    ).iloc[0]

    balanced_pool = sweep[sweep["false_positive_rate"] <= balanced_max_fpr]
    if balanced_pool.empty:
        balanced = sweep.sort_values(sort_cols, ascending=[False, False, True]).iloc[0]
        balanced_feasible = False
    else:
        balanced = balanced_pool.sort_values(sort_cols, ascending=[False, False, True]).iloc[0]
        balanced_feasible = True

    conservative_pool = sweep[sweep["false_positive_rate"] <= conservative_max_fpr]
    if conservative_pool.empty:
        conservative = sweep.sort_values(
            ["recall", "recovery_rate_percent", "false_positive_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        conservative_feasible = False
    else:
        conservative = conservative_pool.sort_values(
            ["recall", "recovery_rate_percent", "false_positive_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        conservative_feasible = True

    ieee_pool = sweep[
        (sweep["recovery_rate_percent"] >= ieee_recovery_min_percent)
        & (sweep["false_positive_rate"] <= balanced_max_fpr)
    ]
    if ieee_pool.empty:
        ieee = balanced
        ieee_feasible = False
    else:
        ieee = ieee_pool.sort_values(sort_cols, ascending=[False, False, True]).iloc[0]
        ieee_feasible = True

    return {
        "Maximum Recovery": max_recovery,
        "Balanced": balanced,
        "Conservative": conservative,
        "IEEE Recommended": ieee,
        "_balanced_feasible": pd.Series({"feasible": balanced_feasible}),
        "_conservative_feasible": pd.Series({"feasible": conservative_feasible}),
        "_ieee_feasible": pd.Series({"feasible": ieee_feasible}),
    }


def _operating_point_row(name: str, row: pd.Series) -> dict[str, Any]:
    return {
        "Configuration Name": name,
        "Recovery Rate": round(float(row["recovery_rate_percent"]), 4),
        "Recall": round(float(row["recall"]), 4),
        "Precision": round(float(row["precision"]), 4),
        "F1": round(float(row["f1"]), 4),
        "FPR": round(float(row["false_positive_rate"]), 4),
        "Recovered Attacks": int(row["recovered_weak_attacks"]),
        "Recovered Benign Windows": int(row.get("recovered_benign_weak_windows", 0)),
        "dbscan_eps": row.get("dbscan_eps"),
        "dbscan_min_samples": row.get("dbscan_min_samples"),
        "minimum_vehicle_count": row.get("minimum_vehicle_count"),
        "minimum_cluster_size": row.get("minimum_cluster_size"),
        "cluster_mean_score_threshold": row.get("cluster_mean_score_threshold"),
        "cluster_cohesion_threshold": row.get("cluster_cohesion_threshold"),
        "strong_support_threshold": row.get("strong_support_threshold"),
    }


def _baseline_from_weak_graph(
    G,
    weak_table: pd.DataFrame,
    cfg: WeakRecoveryOptimizationConfig,
) -> dict[str, float]:
    """Connected-component recovery on a weak-only graph (Top-K or cross-vehicle kNN)."""
    weak_df, clusters, _ = analyse_weak_clusters(G, weak_table)
    recoverable = identify_recoverable(
        weak_df,
        clusters,
        minimum_cluster_size=cfg.weak_minimum_cluster_size,
        minimum_vehicle_count=cfg.weak_minimum_vehicle_count,
        recovery_score_threshold=cfg.recovery_score_threshold,
    )
    return evaluate_selective_promotion(weak_df, recoverable)


def _evaluate_weak_graph_baselines(
    weak_df_full: pd.DataFrame,
    weak_table: pd.DataFrame,
    cfg: WeakRecoveryOptimizationConfig,
) -> dict[str, dict[str, float]]:
    """Original Top-K and Cross-Vehicle kNN on weak-only graphs."""
    G_orig = _build_weak_graph_original(
        weak_df_full,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
    )
    G_cv = _build_weak_graph_cross_vehicle(
        weak_df_full,
        top_k_same=cfg.top_k_same_vehicle,
        top_k_cross=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
    )
    return {
        "Original Top-K": _baseline_from_weak_graph(G_orig, weak_table, cfg),
        "Cross-Vehicle kNN": _baseline_from_weak_graph(G_cv, weak_table, cfg),
    }


def _pareto_frontier(sweep: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated points maximizing recovery while minimizing FPR."""
    pts = sweep[["recovery_rate_percent", "false_positive_rate", "f1"]].copy()
    pts = pts.sort_values(
        ["recovery_rate_percent", "false_positive_rate"],
        ascending=[False, True],
    )
    frontier_rows: list[int] = []
    best_fpr = float("inf")
    for idx, row in pts.iterrows():
        fpr = float(row["false_positive_rate"])
        if fpr <= best_fpr:
            frontier_rows.append(idx)
            best_fpr = fpr
    return sweep.loc[frontier_rows]


def _write_summary(
    path: Path,
    *,
    sweep: pd.DataFrame,
    operating: dict[str, pd.Series],
    feasibility: dict[str, bool],
    baselines: dict[str, dict[str, float]],
    cfg: WeakRecoveryOptimizationConfig,
) -> None:
    max_rec = operating["Maximum Recovery"]
    balanced = operating["Balanced"]
    ieee = operating["IEEE Recommended"]
    ieee_ok = feasibility["ieee"]
    max_recovery_achieved = float(sweep["recovery_rate_percent"].max())
    any_dbscan_recovery = max_recovery_achieved > 0
    n_dbscan_pairs = len(cfg.dbscan_eps_values) * len(cfg.dbscan_min_samples_values)
    eps_lo = min(cfg.dbscan_eps_values)
    eps_hi = max(cfg.dbscan_eps_values)

    q1 = (
        f"**Yes, partially.** Under DBSCAN sub-cluster promotion on the cross-vehicle constrained "
        f"graph, the sweep recovered up to **{max_recovery_achieved:.2f}%** of locally missed "
        f"weak attacks (maximum over {len(sweep):,} configurations). "
        if any_dbscan_recovery
        else (
            f"**No meaningful DBSCAN recovery** under eps grid [{eps_lo:.2f}, {eps_hi:.2f}]: "
            f"all {n_dbscan_pairs} DBSCAN runs assigned every node to noise, yielding zero eligible "
            f"promotion clusters. "
        )
    )
    q1 += (
        f"Original Top-K weak-only recovery: **{baselines['Original Top-K']['recovery_rate_percent']:.2f}%**; "
        f"Cross-Vehicle kNN weak-only: **{baselines['Cross-Vehicle kNN']['recovery_rate_percent']:.2f}%** "
        f"(the latter promotes all weak nodes in one connected component — "
        f"FPR={baselines['Cross-Vehicle kNN']['false_positive_rate']:.4f})."
    )

    lines = [
        "# Final Weak Anomaly Recovery — Summary",
        "",
        "## Research questions",
        "",
        "### 1. Can fleet correlation recover weak anomalies?",
        q1,
        "",
        "### 2. Maximum achievable recovery?",
        f"- **{max_rec['recovery_rate_percent']:.2f}%** at DBSCAN eps={max_rec['dbscan_eps']}, "
        f"min_samples={int(max_rec['dbscan_min_samples'])}, "
        f"mean score ≥ {max_rec['cluster_mean_score_threshold']:.2f}, "
        f"strong support ≥ {max_rec['strong_support_threshold']:.0%} "
        f"(FPR={max_rec['false_positive_rate']:.4f}, F1={max_rec['f1']:.4f}).",
        "",
        "### 3. Best balanced operating point?",
        (
            f"- **Balanced** (FPR ≤ {cfg.balanced_max_fpr:.0%}): recovery "
            f"{balanced['recovery_rate_percent']:.2f}%, F1={balanced['f1']:.4f}, "
            f"FPR={balanced['false_positive_rate']:.4f} "
            f"{'(constraint satisfied)' if feasibility['balanced'] else '(constraint infeasible — best-effort pick)'}."
        ),
        "",
        "### 4. Recommended deployment operating point?",
        (
            f"- **Paper operating point: IEEE Recommended** — "
            f"recovery ≥ {cfg.ieee_recovery_min_percent:.0f}% and FPR ≤ {cfg.balanced_max_fpr:.0%}, "
            f"maximize F1. "
            + (
                f"Selected: recovery {ieee['recovery_rate_percent']:.2f}%, "
                f"F1={ieee['f1']:.4f}, FPR={ieee['false_positive_rate']:.4f} "
                f"(eps={ieee['dbscan_eps']}, min_samples={int(ieee['dbscan_min_samples'])}, "
                f"mean ≥ {ieee['cluster_mean_score_threshold']:.2f}, cohesion ≥ {ieee['cluster_cohesion_threshold']:.2f}, "
                f"strong ≥ {ieee['strong_support_threshold']:.0%})."
                if ieee_ok
                else (
                    f"**IEEE constraints are infeasible** on this dataset "
                    f"(no config with recovery ≥ {cfg.ieee_recovery_min_percent:.0f}% and FPR ≤ {cfg.balanced_max_fpr:.0%}). "
                    f"**Deployment fallback: Balanced operating point** — recovery "
                    f"{balanced['recovery_rate_percent']:.2f}%, F1={balanced['f1']:.4f}, "
                    f"FPR={balanced['false_positive_rate']:.4f} "
                    f"(eps={balanced['dbscan_eps']}, min_samples={int(balanced['dbscan_min_samples'])}, "
                    f"mean ≥ {balanced['cluster_mean_score_threshold']:.2f}, cohesion ≥ "
                    f"{balanced['cluster_cohesion_threshold']:.2f}, strong ≥ {balanced['strong_support_threshold']:.0%})."
                )
            )
        ),
        "",
        "### 5. Trade-off between recovery and false positives?",
        (
            f"Higher recovery generally increases FPR. Across {len(sweep):,} configurations, "
            f"recovery spans [{sweep['recovery_rate_percent'].min():.2f}%, {sweep['recovery_rate_percent'].max():.2f}%] "
            f"and FPR spans [{sweep['false_positive_rate'].min():.4f}, {sweep['false_positive_rate'].max():.4f}]. "
            "See `figures/recovery_vs_fpr_curve.pdf` and `figures/weak_recovery_pareto_frontier.pdf`."
        ),
        "",
        "## Grid search",
        f"- DBSCAN (eps × min_samples): {len(cfg.dbscan_eps_values)} × {len(cfg.dbscan_min_samples_values)} = "
        f"{n_dbscan_pairs} clusterings",
        f"- Promotion configs per clustering: "
        f"{len(cfg.minimum_vehicle_count_values) * len(cfg.minimum_cluster_size_values) * len(cfg.cluster_mean_score_threshold_values) * len(cfg.cluster_cohesion_threshold_values) * len(cfg.strong_support_threshold_values):,}",
        f"- Total evaluated: {len(sweep):,}",
        "",
        "## DBSCAN eps scale",
        "DBSCAN runs in scaled 8-D PCA space (Euclidean distance). Values ≤ 0.20 assign all nodes to noise; "
        "operational clusters appear at eps ≥ 0.90 (see `selective_weak_promotion`, eps=1.2).",
        "",
        "## Graph construction",
        f"- Cross-vehicle constrained kNN ({cfg.top_k_same_vehicle} same + {cfg.top_k_cross_vehicle} cross, τ={cfg.similarity_threshold})",
        "- Clustering: DBSCAN on all anomaly descriptors (weak + strong)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_weak_recovery_optimization(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: WeakRecoveryOptimizationOutputs,
    cfg: WeakRecoveryOptimizationConfig,
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

    build_cross_vehicle_constrained_graph(
        descriptors,
        top_k_same_vehicle=cfg.top_k_same_vehicle,
        top_k_cross_vehicle=cfg.top_k_cross_vehicle,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )

    desc = descriptors.reset_index(drop=True)
    X, _ = resolve_fleet_similarity_matrix(
        desc,
        similarity_feature_view=cfg.similarity_feature_view,
        feature_dominance_threshold=cfg.feature_dominance_threshold,
        allowed_high_dominance_features=cfg.allowed_high_dominance_features,
    )

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

    expected_configs = (
        len(cfg.dbscan_eps_values)
        * len(cfg.dbscan_min_samples_values)
        * len(cfg.minimum_vehicle_count_values)
        * len(cfg.minimum_cluster_size_values)
        * len(cfg.cluster_mean_score_threshold_values)
        * len(cfg.cluster_cohesion_threshold_values)
        * len(cfg.strong_support_threshold_values)
    )
    logger.info("Expected promotion sweep size: %d configurations", expected_configs)

    all_rows: list[dict[str, Any]] = []
    n_dbscan_pairs = len(cfg.dbscan_eps_values) * len(cfg.dbscan_min_samples_values)
    sweep_path = outputs.results_dir / "weak_recovery_parameter_sweep.csv"
    for pair_idx, (eps, min_samples) in enumerate(
        product(cfg.dbscan_eps_values, cfg.dbscan_min_samples_values),
        start=1,
    ):
        logger.info(
            "DBSCAN clustering %d/%d: eps=%.2f, min_samples=%d",
            pair_idx,
            n_dbscan_pairs,
            eps,
            min_samples,
        )
        labels = run_dbscan_on_graph_nodes(
            desc,
            X,
            dbscan_eps=eps,
            dbscan_min_samples=min_samples,
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
        eid_to_cluster = dict(zip(desc["event_id"].astype(str), labels.astype(int)))
        weak_work = weak_df.copy()
        weak_work["cluster_id"] = (
            weak_work["descriptor_id"].astype(str).map(eid_to_cluster).fillna(-1).astype(int)
        )
        all_rows.extend(
            _sweep_promotion_params(
                dbscan_eps=eps,
                dbscan_min_samples=min_samples,
                cluster_stats=cluster_stats,
                weak_df=weak_work,
                min_vehicle_values=cfg.minimum_vehicle_count_values,
                min_cluster_values=cfg.minimum_cluster_size_values,
                mean_score_values=cfg.cluster_mean_score_threshold_values,
                cohesion_values=cfg.cluster_cohesion_threshold_values,
                strong_support_values=cfg.strong_support_threshold_values,
            )
        )
        pd.DataFrame(all_rows).to_csv(sweep_path, index=False)
        logger.info(
            "Checkpoint: %d/%d DBSCAN pairs, %d sweep rows",
            pair_idx,
            n_dbscan_pairs,
            len(all_rows),
        )

    sweep = pd.DataFrame(all_rows)
    sweep.to_csv(sweep_path, index=False)
    logger.info("Wrote %d sweep rows to %s", len(sweep), sweep_path)

    op_raw = _select_operating_points(
        sweep,
        ieee_recovery_min_percent=cfg.ieee_recovery_min_percent,
        balanced_max_fpr=cfg.balanced_max_fpr,
        conservative_max_fpr=cfg.conservative_max_fpr,
    )
    feasibility = {
        "balanced": bool(op_raw["_balanced_feasible"]["feasible"]),
        "conservative": bool(op_raw["_conservative_feasible"]["feasible"]),
        "ieee": bool(op_raw["_ieee_feasible"]["feasible"]),
    }
    operating_points = {
        k: v for k, v in op_raw.items() if not k.startswith("_")
    }

    best_rows = [_operating_point_row(name, row) for name, row in operating_points.items()]
    best_df = pd.DataFrame(best_rows)
    best_path = outputs.results_dir / "weak_recovery_best_configurations.csv"
    best_df.to_csv(best_path, index=False)

    baselines = _evaluate_weak_graph_baselines(weak_full, weak_df, cfg)

    ieee_table_rows = [
        {
            "Configuration": "Original Top-K",
            "Recovery Rate": baselines["Original Top-K"]["recovery_rate_percent"],
            "Recall": baselines["Original Top-K"]["recall"],
            "Precision": baselines["Original Top-K"]["precision"],
            "F1-score": baselines["Original Top-K"]["f1"],
            "FPR": baselines["Original Top-K"]["false_positive_rate"],
        },
        {
            "Configuration": "Cross-Vehicle kNN",
            "Recovery Rate": baselines["Cross-Vehicle kNN"]["recovery_rate_percent"],
            "Recall": baselines["Cross-Vehicle kNN"]["recall"],
            "Precision": baselines["Cross-Vehicle kNN"]["precision"],
            "F1-score": baselines["Cross-Vehicle kNN"]["f1"],
            "FPR": baselines["Cross-Vehicle kNN"]["false_positive_rate"],
        },
    ]
    for label, key in [
        ("DBSCAN Conservative", "Conservative"),
        ("DBSCAN Balanced", "Balanced"),
        ("DBSCAN IEEE Recommended", "IEEE Recommended"),
    ]:
        row = operating_points[key]
        ieee_table_rows.append(
            {
                "Configuration": label,
                "Recovery Rate": float(row["recovery_rate_percent"]),
                "Recall": float(row["recall"]),
                "Precision": float(row["precision"]),
                "F1-score": float(row["f1"]),
                "FPR": float(row["false_positive_rate"]),
            }
        )
    ieee_table = pd.DataFrame(ieee_table_rows).round(4)

    table_tex_path = outputs.tables_dir / "table_weak_recovery_operating_points.tex"
    table_tex_path.write_text(
        _df_to_ieee_tex(
            ieee_table,
            "Weak Anomaly Recovery Performance",
            "tab:weak-recovery-operating-points",
        ),
        encoding="utf-8",
    )
    table_md_path = outputs.tables_dir / "table_weak_recovery_operating_points.md"
    cols = list(ieee_table.columns)
    md_lines = [
        "# Weak Anomaly Recovery Performance",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, r in ieee_table.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    table_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Figure 1: Recovery vs FPR scatter
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        sweep["false_positive_rate"],
        sweep["recovery_rate_percent"],
        s=4,
        alpha=0.25,
        c="#4472C4",
        edgecolors="none",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Weak recovery rate (%)")
    ax.set_title("Recovery vs FPR — parameter sweep")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "recovery_vs_fpr_curve")

    # Figure 2: F1 heatmap (max F1 over other params per cell)
    heat = (
        sweep.groupby(
            ["cluster_mean_score_threshold", "strong_support_threshold"],
            as_index=False,
        )["f1"]
        .max()
        .pivot(
            index="cluster_mean_score_threshold",
            columns="strong_support_threshold",
            values="f1",
        )
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
    ax.set_title("Max F1 — mean score vs strong support")
    ax.set_xlabel("Strong support threshold")
    ax.set_ylabel("Cluster mean score threshold")
    _save_figure(fig, outputs.figures_dir / "weak_recovery_f1_heatmap")

    # Figure 3: Pareto frontier with IEEE point
    frontier = _pareto_frontier(sweep)
    ieee_pt = operating_points["IEEE Recommended"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        sweep["false_positive_rate"],
        sweep["recovery_rate_percent"],
        s=3,
        alpha=0.15,
        c="#B0B0B0",
        edgecolors="none",
        label="All configs",
    )
    ax.plot(
        frontier["false_positive_rate"],
        frontier["recovery_rate_percent"],
        color="#4472C4",
        linewidth=2,
        label="Pareto frontier",
    )
    ax.scatter(
        [ieee_pt["false_positive_rate"]],
        [ieee_pt["recovery_rate_percent"]],
        s=120,
        c="#C00000",
        marker="*",
        zorder=5,
        label="IEEE recommended",
    )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Weak recovery rate (%)")
    ax.set_title("Recovery–FPR Pareto frontier")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "weak_recovery_pareto_frontier")

    summary_path = outputs.results_dir / "final_weak_recovery_summary.md"
    _write_summary(
        summary_path,
        sweep=sweep,
        operating=operating_points,
        feasibility=feasibility,
        baselines=baselines,
        cfg=cfg,
    )

    logger.info(
        "Optimization complete: %d configs, max recovery %.2f%%, IEEE feasible=%s",
        len(sweep),
        sweep["recovery_rate_percent"].max(),
        feasibility["ieee"],
    )
    return {
        "sweep": sweep_path,
        "best_configurations": best_path,
        "summary": summary_path,
        "operating_points_table_tex": table_tex_path,
        "operating_points_table_md": table_md_path,
        "recovery_vs_fpr": outputs.figures_dir / "recovery_vs_fpr_curve.pdf",
        "f1_heatmap": outputs.figures_dir / "weak_recovery_f1_heatmap.pdf",
        "pareto": outputs.figures_dir / "weak_recovery_pareto_frontier.pdf",
    }
