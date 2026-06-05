#!/usr/bin/env python3
"""
Cross-vehicle descriptor generalisation experiment.

Usage:
  python3 run_cross_vehicle_generalisation.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.cross_vehicle_generalisation_experiment import (
    CrossVehicleGeneralisationConfig,
    CrossVehicleGeneralisationOutputs,
    run_cross_vehicle_generalisation_experiment,
)
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_cross_vehicle_generalisation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-vehicle descriptor generalisation")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    sp_cfg = config.get("selective_weak_promotion", {})
    cv_cfg = config.get("cross_vehicle_generalisation", {})

    cfg = CrossVehicleGeneralisationConfig(
        max_embedding_samples=int(cv_cfg.get("max_embedding_samples", 5000)),
        max_similarity_pairs=int(cv_cfg.get("max_similarity_pairs", 30000)),
        dbscan_eps=float(cv_cfg.get("dbscan_eps", sp_cfg.get("dbscan_eps", 1.2))),
        dbscan_min_samples=int(cv_cfg.get("dbscan_min_samples", sp_cfg.get("dbscan_min_samples", 10))),
        dbscan_pca_components=int(cv_cfg.get("dbscan_pca_components", 8)),
        random_forest_estimators=int(cv_cfg.get("random_forest_estimators", 200)),
        logistic_max_iter=int(cv_cfg.get("logistic_max_iter", 2000)),
        seed=int(get_nested(config, "project", "seed", default=42)),
    )

    descriptors_path = paths.root / artifacts.get(
        "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
    )
    features_path = paths.root / artifacts.get(
        "window_features", "data/processed/window_features.csv"
    )

    if not descriptors_path.exists():
        logger.error("Missing anomaly_descriptors: %s", descriptors_path)
        return 1

    outputs = CrossVehicleGeneralisationOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_cross_vehicle_generalisation_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Cross-vehicle generalisation experiment completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
