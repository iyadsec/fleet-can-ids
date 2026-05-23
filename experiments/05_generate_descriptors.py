#!/usr/bin/env python3
"""Generate compact anomaly descriptors from features and IDS predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.descriptor_generator import (
    generate_anomaly_descriptors,
    load_or_generate_predictions,
    print_descriptor_summary,
    save_anomaly_descriptors,
)
from src.models.vehicle_ids import load_feature_dataset
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anomaly descriptors")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/window_features.csv",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/metrics/window_predictions.csv",
        help="Cached per-window IDS predictions (generated if missing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/anomaly_descriptors.csv",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum anomaly score to flag a window",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="random_forest",
        help="Model providing the reported anomaly_score",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--regenerate-predictions",
        action="store_true",
        help="Re-run IDS models even if predictions file exists",
    )
    parser.add_argument(
        "--no-autoencoder",
        action="store_true",
        help="Skip autoencoder when generating predictions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.05_generate_descriptors")
    paths = ProjectPaths.from_root(_ROOT)

    seed = args.seed if args.seed is not None else 42
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            seed = int(cfg.get("project", {}).get("seed", seed))
        except FileNotFoundError:
            logger.warning("Config not found; using default seed.")

    features_path = paths.root / args.features
    predictions_path = paths.root / args.predictions
    output_path = paths.root / args.output

    if not features_path.exists():
        logger.error("Features not found: %s (run 03_extract_features.py)", features_path)
        return 1

    logger.info("Features:    %s", features_path)
    logger.info("Predictions: %s", predictions_path)
    logger.info("Output:      %s", output_path)

    features = load_feature_dataset(features_path)
    predictions = load_or_generate_predictions(
        features,
        predictions_path,
        random_state=seed,
        test_size=args.test_size,
        include_autoencoder=not args.no_autoencoder,
        regenerate=args.regenerate_predictions,
    )

    descriptors = generate_anomaly_descriptors(
        features,
        predictions,
        primary_model=args.primary_model,
        score_threshold=args.score_threshold,
    )

    if descriptors.empty:
        logger.error("No anomaly descriptors generated.")
        return 1

    save_anomaly_descriptors(descriptors, output_path)
    print_descriptor_summary(descriptors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
