#!/usr/bin/env python3
"""Consolidate headerless OCS Lab Car-Hacking release CAN logs into clean_can_release.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.release_dataset_loader import (
    consolidate_release_dataset,
    print_release_consolidation_summary,
    save_clean_release_dataset,
    save_release_summaries,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate headerless release CAN logs to clean_can_release.csv"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--release-root",
        type=str,
        default="data/raw/release",
        help="Root folder containing release CSV files (scanned recursively)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clean_can_release.csv",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.create_clean_can_release")
    paths = ProjectPaths.from_root(_ROOT)

    release_root = paths.root / args.release_root
    output_path = paths.root / args.output
    metrics_dir = paths.root / args.metrics_dir

    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            rel_cfg = cfg.get("release", {})
            if args.release_root == "data/raw/release" and rel_cfg.get("raw_dir"):
                release_root = paths.root / rel_cfg["raw_dir"]
            if args.output == "data/processed/clean_can_release.csv" and rel_cfg.get("output"):
                output_path = paths.root / rel_cfg["output"]
            if rel_cfg.get("metrics_dir"):
                metrics_dir = paths.root / rel_cfg["metrics_dir"]
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    logger.info("Release root: %s", release_root)
    logger.info("Output:       %s", output_path)

    if not release_root.exists():
        logger.error("Release folder not found: %s", release_root)
        logger.error("Expected headerless CSV files under data/raw/release/ (recursive scan).")
        return 1

    result = consolidate_release_dataset(release_root)
    if result.dataframe.empty:
        logger.error("No release frames consolidated. Check files under %s", release_root)
        save_release_summaries(result, metrics_dir)
        return 1

    save_clean_release_dataset(result.dataframe, output_path)
    save_release_summaries(result, metrics_dir)
    print_release_consolidation_summary(result)

    print("clean_can_release.csv successfully generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
