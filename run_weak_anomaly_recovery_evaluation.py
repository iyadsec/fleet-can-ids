#!/usr/bin/env python3
"""
Weak anomaly recovery evaluation (local IDS vs fleet behavioural correlation).

Usage:
  python run_weak_anomaly_recovery_evaluation.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.weak_anomaly_recovery_experiment import (
    WeakRecoveryConfig,
    WeakRecoveryOutputs,
    run_weak_anomaly_recovery_experiment,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_weak_anomaly_recovery_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weak anomaly recovery evaluation")
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
    wr_cfg = config.get("weak_anomaly_recovery", {})
    fg = parse_fleet_graph_similarity_settings(config)

    cfg = WeakRecoveryConfig(
        strong_threshold=float(wr_cfg.get("strong_threshold", ids_cfg.get("strong_threshold", 0.80))),
        weak_threshold=float(wr_cfg.get("weak_threshold", ids_cfg.get("weak_threshold", 0.55))),
        top_k_neighbors=int(wr_cfg.get("top_k_neighbors", fleet_cfg.get("top_k_neighbors", 15))),
        similarity_threshold=float(
            wr_cfg.get("similarity_threshold", fleet_cfg.get("similarity_threshold", 0.95))
        ),
        minimum_cluster_size=int(
            wr_cfg.get("minimum_cluster_size", fleet_cfg.get("minimum_cluster_size", 2))
        ),
        minimum_vehicle_count=int(
            wr_cfg.get("minimum_vehicle_count", fleet_cfg.get("minimum_vehicle_count", 2))
        ),
        recovery_score_threshold=float(
            wr_cfg.get(
                "recovery_score_threshold",
                wr_cfg.get("weak_threshold", ids_cfg.get("weak_threshold", 0.55)),
            )
        ),
        max_viz_nodes=int(
            wr_cfg.get("max_viz_nodes", config.get("research", {}).get("graph_viz_max_nodes", 1200))
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

    outputs = WeakRecoveryOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_weak_anomaly_recovery_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Weak anomaly recovery evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
