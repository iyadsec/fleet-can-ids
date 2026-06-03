#!/usr/bin/env python3
"""
Behaviour-focused fleet graph similarity ablation and validation.

Usage:
  python3 run_behavior_view_similarity.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.behavior_view_similarity_experiment import (
    BehaviorViewConfig,
    BehaviorViewOutputs,
    run_behavior_view_similarity_experiment,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_behavior_view_similarity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behaviour-view fleet similarity ablation")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    fg = parse_fleet_graph_similarity_settings(config)
    fg_cfg = config.get("fleet_graph", {})
    ids_cfg = config.get("vehicle_ids", {})
    fc_cfg = config.get("fleet_correlation", {})
    wr_cfg = config.get("weak_anomaly_recovery", {})
    sp_cfg = config.get("selective_weak_promotion", {})
    bv_cfg = config.get("behavior_view_similarity", {})

    cfg = BehaviorViewConfig(
        top_k_neighbors=int(fg.get("top_k_neighbors", fc_cfg.get("top_k_neighbors", 15))),
        top_k_same_vehicle=int(fg.get("top_k_same_vehicle", sp_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(fg.get("top_k_cross_vehicle", sp_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(fg.get("similarity_threshold", fc_cfg.get("similarity_threshold", 0.95))),
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        allowed_high_dominance_features=fg["allowed_high_dominance_features"],
        strong_threshold=float(ids_cfg.get("strong_threshold", 0.80)),
        weak_threshold=float(ids_cfg.get("weak_threshold", 0.55)),
        fleet_minimum_vehicle_count=int(fc_cfg.get("minimum_vehicle_count", 3)),
        fleet_minimum_cluster_size=int(fc_cfg.get("minimum_cluster_size", 2)),
        fleet_cluster_score_threshold=float(fc_cfg.get("fleet_cluster_score_threshold", 0.7)),
        weak_minimum_vehicle_count=int(wr_cfg.get("minimum_vehicle_count", 2)),
        weak_minimum_cluster_size=int(wr_cfg.get("minimum_cluster_size", 2)),
        recovery_score_threshold=float(wr_cfg.get("recovery_score_threshold", 0.55)),
        dbscan_eps=float(sp_cfg.get("dbscan_eps", 1.2)),
        dbscan_min_samples=int(sp_cfg.get("dbscan_min_samples", 10)),
        dbscan_pca_components=int(sp_cfg.get("dbscan_pca_components", 8)),
        max_clustering_samples=int(sp_cfg.get("max_clustering_samples", 20000)),
        promotion_mean_score=float(bv_cfg.get("promotion_mean_score", 0.60)),
        promotion_strong_support=float(bv_cfg.get("promotion_strong_support", 0.40)),
        seed=int(get_nested(config, "project", "seed", default=42)),
        paper_similarity_view=fg_cfg.get(
            "similarity_feature_view", "behavior_only_vehicle_normalized"
        ),
    )

    descriptors_path = paths.root / artifacts.get(
        "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
    )
    features_path = paths.root / artifacts.get(
        "window_features", "data/processed/window_features.csv"
    )
    cluster_results_path = paths.root / artifacts.get(
        "fleet_cluster_results", "data/processed/fleet_cluster_results.csv"
    )

    outputs = BehaviorViewOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_behavior_view_similarity_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
        cluster_results_path=cluster_results_path if cluster_results_path.exists() else None,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Behaviour-view similarity experiment completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
