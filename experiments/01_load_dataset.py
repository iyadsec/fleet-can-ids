#!/usr/bin/env python3
"""Load external and local CAN datasets; write merged clean CSV and print stats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset_loader import (
    DEFAULT_EXTERNAL_DATASET,
    load_and_merge,
    print_dataset_statistics,
    save_clean_dataset,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and merge CAN bus datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config (optional path overrides)",
    )
    parser.add_argument(
        "--external-root",
        type=str,
        default=None,
        help=f"External dataset root (default: {DEFAULT_EXTERNAL_DATASET})",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Local raw data root (default: data/raw from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/clean_can_data.csv",
        help="Merged output CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.01_load_dataset")
    paths = ProjectPaths.from_root(_ROOT)

    external_root = Path(args.external_root) if args.external_root else DEFAULT_EXTERNAL_DATASET
    raw_root = Path(args.raw_root) if args.raw_root else paths.raw_dir
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = paths.root / output_path

    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            data_cfg = cfg.get("data", {})
            external_root = Path(data_cfg.get("external_dataset_dir", external_root))
            if args.raw_root is None:
                raw_root = paths.root / cfg.get("paths", {}).get("raw_dir", raw_root)
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    logger.info("External dataset root: %s", external_root)
    logger.info("Local raw root:        %s", raw_root)
    logger.info("Output:                %s", output_path)

    df = load_and_merge(external_root=external_root, raw_root=raw_root)
    if df.empty:
        logger.error("No data loaded. Check dataset paths and file layout.")
        return 1

    save_clean_dataset(df, output_path)
    print_dataset_statistics(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
