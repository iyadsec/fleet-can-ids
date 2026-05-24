#!/usr/bin/env python3
"""Stage 1: validate consolidated CAN train dataset before modelling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset_loader import DEFAULT_EXTERNAL_DATASET
from src.data.stage1_consolidated_validation import run_stage1_validation, print_stage1_verdict
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 — consolidated dataset statistics and integrity validation"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/clean_can_train.csv",
        help="Consolidated train CSV",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Directory for validation outputs",
    )
    parser.add_argument("--raw-root", type=str, default=None, help="Local raw CAN logs")
    parser.add_argument(
        "--external-root",
        type=str,
        default=None,
        help="External dataset root for raw row-count check",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only scan data/raw (skip external dataset)",
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default="data/processed/clean_can_data.csv",
        help="Use this file if train CSV is missing (optional)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back to clean_can_data.csv",
    )
    parser.add_argument(
        "--verify-raw",
        action="store_true",
        help="Compare consolidated rows to parseable raw log totals (slow)",
    )
    parser.add_argument(
        "--full-duplicates",
        action="store_true",
        help="Full-file duplicate scan (slow on multi-GB CSVs)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.stage1_validate_consolidated_dataset")
    paths = ProjectPaths.from_root(_ROOT)

    raw_root = Path(args.raw_root) if args.raw_root else paths.raw_dir
    external_root: Path | None = None
    if not args.raw_only:
        external_root = Path(args.external_root) if args.external_root else DEFAULT_EXTERNAL_DATASET

    input_rel = args.input
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            stage1 = cfg.get("stage1", {})
            if input_rel == "data/processed/clean_can_train.csv" and stage1.get("consolidated_train"):
                input_rel = stage1["consolidated_train"]
            if args.raw_root is None:
                raw_root = paths.root / cfg.get("paths", {}).get("raw_dir", raw_root)
            if not args.raw_only and args.external_root is None:
                ext = cfg.get("data", {}).get("external_dataset_dir")
                if ext:
                    external_root = Path(ext)
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    consolidated = paths.root / input_rel
    if not consolidated.exists() and not args.no_fallback:
        fallback = paths.root / args.fallback
        if fallback.exists():
            logger.warning("Train CSV missing; using fallback %s", fallback)
            consolidated = fallback

    raw_roots: list[Path] = [raw_root]
    if external_root is not None and external_root.exists():
        raw_roots.append(external_root)

    metrics_dir = paths.root / args.metrics_dir
    logger.info("Input:       %s", consolidated)
    logger.info("Metrics dir: %s", metrics_dir)
    logger.info("Raw roots:   %s", raw_roots)

    verify_raw = args.verify_raw
    if args.config and not verify_raw:
        try:
            cfg = load_config(paths.root / args.config)
            verify_raw = bool(cfg.get("stage1", {}).get("verify_raw_rows", False))
        except FileNotFoundError:
            pass

    report = run_stage1_validation(
        consolidated,
        raw_roots,
        metrics_dir=metrics_dir,
        verify_raw_rows=verify_raw,
        full_duplicates=args.full_duplicates,
    )
    print_stage1_verdict(report)

    if report.artifacts:
        print("Outputs written:")
        for name, p in report.artifacts.items():
            print(f"  {name}: {p}")

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
