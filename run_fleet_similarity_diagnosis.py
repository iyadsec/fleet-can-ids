#!/usr/bin/env python3
"""
Diagnose fleet graph cross-vehicle similarity failures.

Usage:
  python run_fleet_similarity_diagnosis.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.fleet_similarity_diagnosis import (
    DiagnosisConfig,
    DiagnosisOutputs,
    run_fleet_similarity_diagnosis,
)
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_fleet_similarity_diagnosis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fleet similarity diagnosis")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    fleet_cfg = config.get("fleet_correlation", {})
    diag_cfg = config.get("fleet_similarity_diagnosis", {})

    cfg = DiagnosisConfig(
        top_k_neighbors=int(diag_cfg.get("top_k_neighbors", fleet_cfg.get("top_k_neighbors", 15))),
        similarity_threshold=float(
            diag_cfg.get("similarity_threshold", fleet_cfg.get("similarity_threshold", 0.95))
        ),
        max_pairs_per_category=int(diag_cfg.get("max_pairs_per_category", 50000)),
        seed=int(get_nested(config, "project", "seed", default=42)),
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

    outputs = DiagnosisOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_fleet_similarity_diagnosis(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    print("Fleet similarity diagnosis completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
