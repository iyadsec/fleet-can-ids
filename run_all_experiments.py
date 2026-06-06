#!/usr/bin/env python3
"""
Generate final research deliverables for the fleet-aware CAN IDS paper.

Runs the existing end-to-end pipeline and then exports publication-ready outputs:

  data loading → preprocessing → window generation → self-supervised training
  → descriptor generation → compression analysis → graph construction → [optional GNN]
  → clustering → evaluation → evidence package export

Usage:
  python run_all_experiments.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.publication_package import export_publication_package
from src.pipeline.full_pipeline import PIPELINE_STEPS, FullPipelineRunner
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all experiments and export publication-ready deliverables"
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/fleet_ids.yaml",
        help="YAML config file (default: configs/fleet_ids.yaml)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose primary output files already exist",
    )
    p.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=PIPELINE_STEPS,
        help="Run only these steps (debug)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("run_all_experiments")
    paths = ProjectPaths.from_root(_ROOT)
    config_path = paths.root / args.config
    config = load_config(config_path)

    pipeline_cfg = config.setdefault("pipeline", {})
    if args.skip_existing:
        pipeline_cfg["skip_if_outputs_exist"] = True

    steps = pipeline_cfg.get("steps", list(PIPELINE_STEPS))
    if args.only:
        steps = list(args.only)

    logger.info("Config: %s", config_path)
    logger.info("Steps:  %s", " → ".join(steps))

    runner = FullPipelineRunner(config, paths)
    code = runner.run(steps)
    if code != 0:
        logger.error("Pipeline failed with exit code %d", code)
        return code

    artifacts = (pipeline_cfg.get("artifacts") or {}).copy()
    written = export_publication_package(
        project_root=paths.root,
        config=config,
        artifacts=artifacts,
    )
    if written:
        logger.info("Exported publication package.")
        for k, v in written.items():
            logger.info("%s: %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

