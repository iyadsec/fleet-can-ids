#!/usr/bin/env python3
"""Train and evaluate vehicle-level intrusion detection models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.vehicle_ids import (
    SELF_SUPERVISED_IDS_MODEL,
    evaluate_vehicle_anomaly_predictions,
    generate_vehicle_anomaly_predictions,
    load_feature_dataset,
    plot_vehicle_confusion_matrices,
    print_results_summary,
    save_vehicle_anomaly_predictions,
    save_results,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle-level CAN intrusion detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/window_features.csv",
        help="Window feature CSV",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/metrics/vehicle_level_self_supervised_results.csv",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/processed/vehicle_anomaly_predictions.csv",
        help="Canonical vehicle IDS predictions for downstream fleet stages",
    )
    parser.add_argument("--primary-model", type=str, default=SELF_SUPERVISED_IDS_MODEL)
    parser.add_argument(
        "--model-output",
        type=str,
        default="outputs/models/vehicle_isolation_forest.joblib",
    )
    parser.add_argument(
        "--confusion-plot",
        type=str,
        default="outputs/figures/confusion_matrix_vehicle.png",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--strong-threshold", type=float, default=None)
    parser.add_argument("--weak-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-autoencoder",
        action="store_true",
        help="Skip PyTorch autoencoder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.04_train_vehicle_ids")
    paths = ProjectPaths.from_root(_ROOT)

    seed = args.seed if args.seed is not None else 42
    strong_threshold = args.strong_threshold if args.strong_threshold is not None else 0.80
    weak_threshold = args.weak_threshold if args.weak_threshold is not None else 0.55
    ids_cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            seed = int(cfg.get("project", {}).get("seed", seed))
            ids_cfg = cfg.get("vehicle_ids", {})
            strong_threshold = float(ids_cfg.get("strong_threshold", strong_threshold))
            weak_threshold = float(ids_cfg.get("weak_threshold", weak_threshold))
        except FileNotFoundError:
            logger.warning("Config not found; using default seed.")

    features_path = paths.root / args.features
    results_path = paths.root / args.results
    predictions_path = paths.root / args.predictions
    confusion_path = paths.root / args.confusion_plot
    model_path = paths.root / args.model_output

    if not features_path.exists():
        logger.error("Features not found: %s (run 03_extract_features.py)", features_path)
        return 1

    logger.info("Features: %s", features_path)
    logger.info("Seed:     %d", seed)

    features = load_feature_dataset(features_path)
    predictions = generate_vehicle_anomaly_predictions(
        features,
        primary_model=args.primary_model,
        test_size=args.test_size,
        random_state=seed,
        include_autoencoder=not args.no_autoencoder,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        model_path=model_path,
    )
    results = evaluate_vehicle_anomaly_predictions(predictions)
    if results.empty:
        logger.error("No model results produced.")
        return 1

    save_results(results, results_path)
    save_vehicle_anomaly_predictions(predictions, predictions_path)
    plot_vehicle_confusion_matrices(results, confusion_path)
    print_results_summary(results)
    print("Vehicle-level IDS updated to self-supervised Isolation Forest and aligned with paper methodology.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
