#!/usr/bin/env python3
"""
Fleet-level correlation improvement experiment.

Compares local-only IDS vs fleet-aware graph correlation using existing
predictions and descriptors (no vehicle IDS retraining).

Usage:
  python run_fleet_correlation_evaluation.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.fleet_correlation_experiment import (
    FleetCorrelationConfig,
    FleetCorrelationOutputs,
    run_fleet_correlation_experiment,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_fleet_correlation_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fleet correlation local vs fleet-aware evaluation")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    graph_cfg = config.get("graph", {})
    final_cfg = config.get("final_decision", {})
    fleet_cfg = config.get("fleet_correlation", {})
    fg = parse_fleet_graph_similarity_settings(config)

    fc = FleetCorrelationConfig(
        top_k_neighbors=int(
            fleet_cfg.get(
                "top_k_neighbors",
                graph_cfg.get("max_neighbors", graph_cfg.get("max_neighbours_per_node", 15)),
            )
        ),
        similarity_threshold=float(fleet_cfg.get("similarity_threshold", 0.95)),
        minimum_cluster_size=int(
            fleet_cfg.get("minimum_cluster_size", final_cfg.get("minimum_cluster_size", 2))
        ),
        minimum_vehicle_count=int(fleet_cfg.get("minimum_vehicle_count", 3)),
        fleet_cluster_score_threshold=float(
            fleet_cfg.get("fleet_cluster_score_threshold", 0.7)
        ),
        max_graph_viz_nodes=int(
            fleet_cfg.get("max_graph_viz_nodes", config.get("research", {}).get("graph_viz_max_nodes", 800))
        ),
        graph_viz_seed=int(get_nested(config, "project", "seed", default=42)),
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

    outputs = FleetCorrelationOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_fleet_correlation_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        cluster_results_path=cluster_results_path if cluster_results_path.exists() else None,
        outputs=outputs,
        cfg=fc,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Fleet correlation evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
