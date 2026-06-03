#!/usr/bin/env python3
"""
Selective weak anomaly promotion with cluster evidence gates and sensitivity analysis.

Usage:
  python run_selective_weak_promotion.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.selective_weak_promotion import (
    SelectivePromotionConfig,
    SelectivePromotionOutputs,
    run_selective_weak_promotion,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_selective_weak_promotion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective weak anomaly promotion sensitivity analysis")
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
    sp_cfg = config.get("selective_weak_promotion", {})
    fg = parse_fleet_graph_similarity_settings(config)

    cluster_cfg = config.get("clustering", {})
    _max_cluster_samples = sp_cfg.get(
        "max_clustering_samples", cluster_cfg.get("max_clustering_samples", 20000)
    )

    cfg = SelectivePromotionConfig(
        strong_threshold=float(sp_cfg.get("strong_threshold", ids_cfg.get("strong_threshold", 0.80))),
        weak_threshold=float(sp_cfg.get("weak_threshold", ids_cfg.get("weak_threshold", 0.55))),
        top_k_same_vehicle=int(sp_cfg.get("top_k_same_vehicle", cmp_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(sp_cfg.get("top_k_cross_vehicle", cmp_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(
            sp_cfg.get("similarity_threshold", cmp_cfg.get("similarity_threshold", 0.95))
        ),
        min_vehicles=int(sp_cfg.get("min_vehicles", 2)),
        min_cluster_size=int(sp_cfg.get("min_cluster_size", 5)),
        min_cohesion=float(sp_cfg.get("min_cohesion", 0.97)),
        mean_score_thresholds=tuple(
            float(x) for x in sp_cfg.get("mean_score_thresholds", [0.60, 0.70, 0.80])
        ),
        strong_support_thresholds=tuple(
            float(x) for x in sp_cfg.get("strong_support_thresholds", [0.40, 0.60, 0.80])
        ),
        meaningful_recovery_min_rate=float(sp_cfg.get("meaningful_recovery_min_rate", 1.0)),
        max_false_positive_rate=float(sp_cfg.get("max_false_positive_rate", 0.50)),
        dbscan_eps=float(sp_cfg.get("dbscan_eps", cluster_cfg.get("dbscan_eps", 1.2))),
        dbscan_min_samples=int(sp_cfg.get("dbscan_min_samples", cluster_cfg.get("dbscan_min_samples", 10))),
        dbscan_pca_components=int(
            sp_cfg.get("dbscan_pca_components", cluster_cfg.get("dbscan_pca_components", 8))
        ),
        max_clustering_samples=int(_max_cluster_samples) if _max_cluster_samples is not None else None,
        max_cohesion_pairs=int(sp_cfg.get("max_cohesion_pairs", 8000)),
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

    outputs = SelectivePromotionOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_selective_weak_promotion(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Selective weak promotion analysis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
