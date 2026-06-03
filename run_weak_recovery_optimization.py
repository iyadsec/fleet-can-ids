#!/usr/bin/env python3
"""
Final weak anomaly recovery optimization — full DBSCAN + promotion grid search.

Usage:
  python3 run_weak_recovery_optimization.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.evaluation.weak_recovery_optimization import (
    DEFAULT_CLUSTER_COHESION_VALUES,
    DEFAULT_CLUSTER_MEAN_SCORE_VALUES,
    DEFAULT_DBSCAN_EPS_VALUES,
    DEFAULT_DBSCAN_MIN_SAMPLES_VALUES,
    DEFAULT_MINIMUM_CLUSTER_SIZE_VALUES,
    DEFAULT_MINIMUM_VEHICLE_COUNT_VALUES,
    DEFAULT_STRONG_SUPPORT_VALUES,
    WeakRecoveryOptimizationConfig,
    WeakRecoveryOptimizationOutputs,
    run_weak_recovery_optimization,
)
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_weak_recovery_optimization")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weak anomaly recovery parameter optimization (DBSCAN + promotion gates)"
    )
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    ids_cfg = config.get("vehicle_ids", {})
    cmp_cfg = config.get("graph_construction_comparison", {})
    opt_cfg = config.get("weak_recovery_optimization", {})
    cluster_cfg = config.get("clustering", {})
    sp_cfg = config.get("selective_weak_promotion", {})
    fg = parse_fleet_graph_similarity_settings(config)

    _max_cluster_samples = opt_cfg.get(
        "max_clustering_samples",
        sp_cfg.get("max_clustering_samples", cluster_cfg.get("max_clustering_samples", 20000)),
    )

    cfg = WeakRecoveryOptimizationConfig(
        strong_threshold=float(opt_cfg.get("strong_threshold", ids_cfg.get("strong_threshold", 0.80))),
        weak_threshold=float(opt_cfg.get("weak_threshold", ids_cfg.get("weak_threshold", 0.55))),
        top_k_same_vehicle=int(opt_cfg.get("top_k_same_vehicle", cmp_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(opt_cfg.get("top_k_cross_vehicle", cmp_cfg.get("top_k_cross_vehicle", 5))),
        top_k_neighbors=int(opt_cfg.get("top_k_neighbors", cmp_cfg.get("top_k_neighbors", 15))),
        similarity_threshold=float(
            opt_cfg.get("similarity_threshold", cmp_cfg.get("similarity_threshold", 0.95))
        ),
        weak_minimum_cluster_size=int(
            opt_cfg.get("weak_minimum_cluster_size", cmp_cfg.get("weak_minimum_cluster_size", 2))
        ),
        weak_minimum_vehicle_count=int(
            opt_cfg.get("weak_minimum_vehicle_count", cmp_cfg.get("weak_minimum_vehicle_count", 2))
        ),
        recovery_score_threshold=float(
            opt_cfg.get("recovery_score_threshold", cmp_cfg.get("recovery_score_threshold", 0.55))
        ),
        dbscan_pca_components=int(
            opt_cfg.get("dbscan_pca_components", cluster_cfg.get("dbscan_pca_components", 8))
        ),
        max_clustering_samples=int(_max_cluster_samples) if _max_cluster_samples is not None else None,
        max_cohesion_pairs=int(opt_cfg.get("max_cohesion_pairs", sp_cfg.get("max_cohesion_pairs", 8000))),
        ieee_recovery_min_percent=float(opt_cfg.get("ieee_recovery_min_percent", 10.0)),
        balanced_max_fpr=float(opt_cfg.get("balanced_max_fpr", 0.10)),
        conservative_max_fpr=float(opt_cfg.get("conservative_max_fpr", 0.05)),
        dbscan_eps_values=tuple(
            float(x) for x in opt_cfg.get("dbscan_eps_values", DEFAULT_DBSCAN_EPS_VALUES)
        ),
        dbscan_min_samples_values=tuple(
            int(x) for x in opt_cfg.get("dbscan_min_samples_values", DEFAULT_DBSCAN_MIN_SAMPLES_VALUES)
        ),
        minimum_vehicle_count_values=tuple(
            int(x) for x in opt_cfg.get("minimum_vehicle_count_values", DEFAULT_MINIMUM_VEHICLE_COUNT_VALUES)
        ),
        minimum_cluster_size_values=tuple(
            int(x) for x in opt_cfg.get("minimum_cluster_size_values", DEFAULT_MINIMUM_CLUSTER_SIZE_VALUES)
        ),
        cluster_mean_score_threshold_values=tuple(
            float(x)
            for x in opt_cfg.get("cluster_mean_score_threshold_values", DEFAULT_CLUSTER_MEAN_SCORE_VALUES)
        ),
        cluster_cohesion_threshold_values=tuple(
            float(x) for x in opt_cfg.get("cluster_cohesion_threshold_values", DEFAULT_CLUSTER_COHESION_VALUES)
        ),
        strong_support_threshold_values=tuple(
            float(x) for x in opt_cfg.get("strong_support_threshold_values", DEFAULT_STRONG_SUPPORT_VALUES)
        ),
        seed=int(get_nested(config, "project", "seed", default=42)),
        similarity_feature_view=fg["similarity_feature_view"],
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        allowed_high_dominance_features=fg["allowed_high_dominance_features"],
    )

    descriptors_path = paths.root / artifacts.get(
        "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
    )
    features_path = paths.root / artifacts.get(
        "window_features", "data/processed/window_features.csv"
    )

    for p, name in [
        (descriptors_path, "anomaly_descriptors"),
        (features_path, "window_features"),
    ]:
        if not p.exists():
            logger.error("Missing %s: %s", name, p)
            return 1

    outputs = WeakRecoveryOptimizationOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_weak_recovery_optimization(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Weak recovery optimization completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
