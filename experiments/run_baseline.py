#!/usr/bin/env python3
"""Deprecated baseline scaffold — use run_all_experiments.py or run_*.py evaluations instead."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deprecated — use run_all_experiments.py")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.run_baseline")
    logger.warning(
        "experiments/run_baseline.py is deprecated. "
        "Vehicle IDS, fleet graphs, and campaign detection are implemented under src/ "
        "and run via run_all_experiments.py or run_*.py (see README.md)."
    )
    logger.info("Config argument ignored: %s", args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
