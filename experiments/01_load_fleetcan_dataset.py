#!/usr/bin/env python3
"""Load fleetcan-sim sessions into clean_can_data.csv (STANDARD_COLUMNS)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset_loader import print_dataset_statistics, save_clean_dataset
from src.data.fleetcan_adapter import (
    dataset_root_from_release,
    default_pilot_sessions,
    load_sessions,
)
from src.utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt fleetcan-sim sessions into pipeline clean CAN CSV"
    )
    parser.add_argument(
        "--release-root",
        type=str,
        default="data/external/fleetcan-sim-scale-1.0",
        help="Restored release directory",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="*",
        default=None,
        help="Explicit session IDs (default: --pilot set)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=True,
        help="Use default N=100 S0–S3 flooding pilot (default: true)",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Load every session under data/raw (large)",
    )
    parser.add_argument(
        "--include-eval-columns",
        action="store_true",
        help="Keep session_id/frame_uid/campaign_id for evaluation joins",
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
    logger = get_logger("experiments.01_load_fleetcan_dataset")
    release = Path(args.release_root)
    if not release.is_absolute():
        release = _ROOT / release
    output = Path(args.output)
    if not output.is_absolute():
        output = _ROOT / output

    try:
        data_root = dataset_root_from_release(release)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.all_sessions:
        sessions = None
    elif args.sessions:
        sessions = args.sessions
    else:
        sessions = default_pilot_sessions()

    logger.info("Data root: %s", data_root)
    logger.info("Sessions:  %s", sessions if sessions is not None else "(all)")
    logger.info("Output:    %s", output)

    df = load_sessions(
        data_root,
        sessions,
        include_eval_columns=args.include_eval_columns,
    )
    if df.empty:
        logger.error("No frames loaded.")
        return 1

    save_clean_dataset(df, output)
    print_dataset_statistics(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
