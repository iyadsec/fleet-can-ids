#!/usr/bin/env python3
"""
Compare original top-k vs cross-vehicle constrained kNN fleet graphs.

Usage:
  python run_graph_construction_comparison.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.graph_construction_comparison import (
    ComparisonConfig,
    ComparisonOutputs,
    run_graph_construction_comparison,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_graph_construction_comparison")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fleet graph construction methods")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    ids_cfg = config.get("vehicle_ids", {})
    fleet_cfg = config.get("fleet_correlation", {})
    weak_cfg = config.get("weak_anomaly_recovery", {})
    cmp_cfg = config.get("graph_construction_comparison", {})
    fg = parse_fleet_graph_similarity_settings(config)

    cfg = ComparisonConfig(
        top_k_neighbors=int(cmp_cfg.get("top_k_neighbors", fleet_cfg.get("top_k_neighbors", 15))),
        top_k_same_vehicle=int(cmp_cfg.get("top_k_same_vehicle", 10)),
        top_k_cross_vehicle=int(cmp_cfg.get("top_k_cross_vehicle", 5)),
        similarity_threshold=float(
            cmp_cfg.get("similarity_threshold", fleet_cfg.get("similarity_threshold", 0.95))
        ),
        strong_threshold=float(ids_cfg.get("strong_threshold", 0.80)),
        weak_threshold=float(ids_cfg.get("weak_threshold", 0.55)),
        fleet_minimum_cluster_size=int(
            cmp_cfg.get("minimum_cluster_size", fleet_cfg.get("minimum_cluster_size", 2))
        ),
        fleet_minimum_vehicle_count=int(
            cmp_cfg.get("fleet_minimum_vehicle_count", fleet_cfg.get("minimum_vehicle_count", 3))
        ),
        fleet_cluster_score_threshold=float(
            cmp_cfg.get(
                "fleet_cluster_score_threshold",
                fleet_cfg.get("fleet_cluster_score_threshold", 0.7),
            )
        ),
        weak_minimum_cluster_size=int(weak_cfg.get("minimum_cluster_size", 2)),
        weak_minimum_vehicle_count=int(weak_cfg.get("minimum_vehicle_count", 2)),
        recovery_score_threshold=float(weak_cfg.get("recovery_score_threshold", 0.55)),
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
    cluster_results_path = paths.root / artifacts.get(
        "fleet_cluster_results", "data/processed/fleet_cluster_results.csv"
    )

    for p, name in [
        (descriptors_path, "anomaly_descriptors"),
        (features_path, "window_features"),
    ]:
        if not p.exists():
            logger.error("Missing %s: %s", name, p)
            return 1

    outputs = ComparisonOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_graph_construction_comparison(
        descriptors_path=descriptors_path,
        features_path=features_path,
        cluster_results_path=cluster_results_path if cluster_results_path.exists() else None,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Graph construction comparison completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
