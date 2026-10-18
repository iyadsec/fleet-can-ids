#!/usr/bin/env python3
"""Validate consolidated CAN dataset against raw sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset_loader import DEFAULT_EXTERNAL_DATASET
from src.data.validate_dataset import (
    ValidationReport,
    load_clean_dataset,
    print_validation_results,
    run_validation,
    save_validation_report,
    validate_byte_range,
    validate_dlc_range,
    validate_duplicates,
    validate_group_counts,
    validate_missing_values,
    validate_random_sample,
    validate_required_columns,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate clean CAN dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config (for raw/external paths)",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Local raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--external-root",
        type=str,
        default=None,
        help="External dataset root (included in row-count check)",
    )
    parser.add_argument(
        "--clean",
        type=str,
        default="data/processed/clean_can_data.csv",
        help="Consolidated clean CAN CSV",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default="outputs/metrics/data_validation_report.csv",
        help="Validation report CSV",
    )
    parser.add_argument(
        "--report-txt",
        type=str,
        default="outputs/metrics/data_validation_summary.txt",
        help="Validation summary text file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Random rows to print for check 7",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only scan data/raw (do not add external dataset root)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Skip raw-vs-clean row counts (for adapted sources like fleetcan-sim)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.01b_validate_clean_dataset")
    paths = ProjectPaths.from_root(_ROOT)

    raw_root = Path(args.raw_root) if args.raw_root else paths.raw_dir
    external_root: Path | None = None
    if not args.raw_only:
        external_root = Path(args.external_root) if args.external_root else DEFAULT_EXTERNAL_DATASET

    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            if args.raw_root is None:
                raw_root = paths.root / cfg.get("paths", {}).get("raw_dir", raw_root)
            if not args.raw_only and args.external_root is None:
                ext = cfg.get("data", {}).get("external_dataset_dir")
                if ext:
                    external_root = Path(ext)
        except FileNotFoundError:
            logger.warning("Config not found; using defaults.")

    clean_path = paths.root / args.clean
    report_csv = paths.root / args.report_csv
    report_txt = paths.root / args.report_txt

    raw_roots: list[Path] = [raw_root]
    if external_root is not None and external_root.exists():
        raw_roots.append(external_root)

    logger.info("Clean dataset: %s", clean_path)
    logger.info("Raw roots:     %s", raw_roots)
    logger.info("Schema only:   %s", args.schema_only)

    if not clean_path.exists():
        logger.error("Clean dataset not found. Run 01_load_dataset.py first.")
        return 1

    if args.schema_only:
        df = load_clean_dataset(clean_path)
        report = ValidationReport()
        validate_required_columns(report, df)
        validate_missing_values(report, df)
        validate_group_counts(report, df, "vehicle_model", "04", "Records per vehicle_model")
        validate_group_counts(report, df, "attack_type", "05", "Records per attack_type")
        validate_group_counts(report, df, "source_file", "06", "Records per source_file")
        validate_random_sample(report, df, n=args.sample_size)
        validate_dlc_range(report, df)
        validate_byte_range(report, df)
        validate_duplicates(report, df)
    else:
        report = run_validation(
            clean_path,
            raw_roots,
            sample_size=args.sample_size,
        )
    save_validation_report(report, report_csv, report_txt)
    print_validation_results(report)
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
