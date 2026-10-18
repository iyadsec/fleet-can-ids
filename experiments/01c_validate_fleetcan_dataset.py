#!/usr/bin/env python3
"""Validate restored fleetcan-sim-scale-1.0 against pipeline schema requirements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.fleetcan_adapter import (
    dataset_root_from_release,
    default_pilot_sessions,
    validate_dataset_coverage,
)
from src.utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fleetcan-sim dataset for FLEET-GUARD pipeline readiness"
    )
    parser.add_argument(
        "--release-root",
        type=str,
        default="data/external/fleetcan-sim-scale-1.0",
        help="Restored release directory (contains data/raw or raw/)",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="*",
        default=None,
        help="Session IDs to check (default: pilot S0–S3 set if present, else all)",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Restrict checks to the default N=20 S0–S3 pilot sessions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metrics/fleetcan_schema_validation.json",
        help="JSON report path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.01c_validate_fleetcan_dataset")
    release = Path(args.release_root)
    if not release.is_absolute():
        release = _ROOT / release

    try:
        data_root = dataset_root_from_release(release)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.pilot:
        sessions = default_pilot_sessions()
    elif args.sessions:
        sessions = args.sessions
    else:
        sessions = None

    logger.info("Data root: %s", data_root)
    report = validate_dataset_coverage(data_root, session_ids=sessions)
    out = Path(args.output)
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)

    n_pass = sum(1 for s in report["sessions"].values() if s["status"] == "PASS")
    n_fail = sum(1 for s in report["sessions"].values() if s["status"] == "FAIL")
    print("\n=== Fleetcan Schema Validation ===")
    print(f"Data root:  {report['data_root']}")
    print(f"Sessions:   {report['n_sessions_checked']} (PASS={n_pass}, FAIL={n_fail})")
    print(f"Overall:    {report['overall']}")
    for sid, entry in list(report["sessions"].items())[:20]:
        print(f"  [{entry['status']}] {sid}")
        for issue in entry.get("issues", [])[:5]:
            print(f"           - {issue}")
    if len(report["sessions"]) > 20:
        print(f"  ... {len(report['sessions']) - 20} more sessions")
    print("==================================\n")

    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
