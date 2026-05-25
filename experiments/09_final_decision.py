#!/usr/bin/env python3
"""Classify anomaly events as isolated or fleet-level coordinated patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.final_decision import (
    classify_final_outcomes,
    load_cluster_results,
    save_final_outcomes,
    summarize_final_outcomes,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final anomaly classifications")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--clusters", type=str, default="data/processed/fleet_cluster_results.csv")
    parser.add_argument(
        "--outcomes",
        type=str,
        default="outputs/metrics/final_detection_outcomes.csv",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/metrics/final_outcome_summary.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.09_final_decision")
    paths = ProjectPaths.from_root(_ROOT)
    cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    decision_cfg = cfg.get("final_decision", {})
    cluster_path = paths.root / args.clusters
    if not cluster_path.exists():
        logger.error("Cluster results not found: %s (run 08_cluster_campaigns.py)", cluster_path)
        return 1

    clusters = load_cluster_results(cluster_path)
    outcomes = classify_final_outcomes(
        clusters,
        similarity_threshold=float(decision_cfg.get("similarity_threshold", 0.85)),
        min_vehicles=int(decision_cfg.get("min_vehicles", 2)),
    )
    summary = summarize_final_outcomes(outcomes)
    save_final_outcomes(
        outcomes,
        summary,
        outcomes_path=paths.root / args.outcomes,
        summary_path=paths.root / args.summary,
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

