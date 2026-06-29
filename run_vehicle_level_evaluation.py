#!/usr/bin/env python3
"""
Generate vehicle-level IDS paper outputs (Table 1 + Figures 1–3).

Usage:
  python run_vehicle_level_evaluation.py --config configs/fleet_ids.yaml

Requires processed window features (run pipeline steps load_dataset → extract_features first).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.vehicle_level_evaluation import EvaluationOutputs, run_vehicle_level_evaluation
from src.models.vehicle_ids import load_feature_dataset
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_vehicle_level_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle-level IDS evaluation for paper outputs")
    p.add_argument(
        "--config",
        type=str,
        default="configs/fleet_ids.yaml",
        help="YAML configuration file",
    )
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="Override path to window_features.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)

    data_cfg = config.get("data", {})
    ids_cfg = config.get("vehicle_ids", {})
    eval_cfg = config.get("vehicle_level_evaluation", {})
    pub = config.get("publication", {})

    features_path = Path(args.features) if args.features else paths.processed_dir / "window_features.csv"
    if not features_path.exists():
        logger.error(
            "Missing %s. Run: python experiments/run_full_pipeline.py "
            "--only load_dataset generate_windows extract_features",
            features_path,
        )
        return 1

    outputs = EvaluationOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    features = load_feature_dataset(features_path)
    seed = int(get_nested(config, "project", "seed", default=42))
    train_ratio = float(eval_cfg.get("train_split", data_cfg.get("train_split", 0.7)))
    val_ratio = float(eval_cfg.get("val_split", data_cfg.get("val_split", 0.15)))
    test_ratio = float(eval_cfg.get("test_split", data_cfg.get("test_split", 0.15)))
    n_estimators = int(eval_cfg.get("n_estimators", ids_cfg.get("n_estimators", 200)))

    logger.info("Loading features from %s (%d windows)", features_path, len(features))
    written = run_vehicle_level_evaluation(
        features,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=seed,
        n_estimators=n_estimators,
        outputs=outputs,
    )
    for name, path in written.items():
        logger.info("%s → %s", name, path)
    print("Vehicle-level IDS evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
