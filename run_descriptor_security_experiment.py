#!/usr/bin/env python3
"""
Descriptor compactness and security enhancement experiment.

Uses existing vehicle IDS predictions and descriptors (no anomaly detector retraining).

Usage:
  python run_descriptor_security_experiment.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.descriptor_security_experiment import SecurityOutputs, run_descriptor_security_experiment
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_descriptor_security_experiment")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descriptor compactness, scalability, and security experiments"
    )
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    sec_cfg = config.get("descriptor_security", {})
    feat_cfg = config.get("features", {})

    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    features_path = paths.root / artifacts.get(
        "window_features", "data/processed/window_features.csv"
    )
    predictions_path = paths.root / artifacts.get(
        "vehicle_anomaly_predictions", "data/processed/vehicle_anomaly_predictions.csv"
    )
    descriptors_path = paths.root / artifacts.get(
        "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
    )
    transmit_path = paths.root / artifacts.get(
        "anomaly_descriptors_transmit", "data/processed/anomaly_descriptors_transmit.csv"
    )

    if not features_path.exists():
        logger.error("Missing window features: %s", features_path)
        return 1

    outputs = SecurityOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    fleet_sizes = tuple(sec_cfg.get("fleet_sizes", [10, 50, 100, 500, 1000]))
    written = run_descriptor_security_experiment(
        features_path=features_path,
        predictions_path=predictions_path,
        descriptors_path=descriptors_path,
        transmit_path=transmit_path,
        outputs=outputs,
        window_size=int(feat_cfg.get("window_size", 100)),
        seed=int(get_nested(config, "project", "seed", default=42)),
        fleet_sizes=fleet_sizes,
    )
    for name, path in written.items():
        logger.info("%s → %s", name, path)
    print("Descriptor compactness and security experiment completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
