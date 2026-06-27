#!/usr/bin/env python3
"""Generate research-ready figures, tables, and experiment_report.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.research_outputs import generate_all_research_outputs
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate research outputs and experiment report")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--regenerate-clustering",
        action="store_true",
        help="Re-run campaign clustering before plotting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.09_generate_research_outputs")
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)

    logger.info("Generating research figures and report...")
    result = generate_all_research_outputs(
        config,
        paths.root,
        regenerate_clustering=args.regenerate_clustering,
    )
    logger.info("Figures: %s", result.get("figures", {}))
    logger.info("Tables:  %s", result.get("tables", {}))
    report = result.get("figures", {}).get("report", "outputs/experiment_report.md")
    logger.info("Report:  %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
