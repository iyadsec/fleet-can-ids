#!/usr/bin/env python3
"""
Run the full Fleet CAN-IDS experiment pipeline.

Conceptual flow:
  Raw CAN data
  → Vehicle anomaly detection
  → Strong/weak anomaly evidence classification
  → Descriptor generation
  → Raw-vs-descriptor size comparison
  → Behavioural graph construction
  → GNN embedding learning
  → Clustering
  → Final outcome classification
  → Research evidence summary and figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pipeline.full_pipeline import PIPELINE_STEPS, FullPipelineRunner
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Fleet CAN-IDS pipeline from YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML configuration file",
    )
    parser.add_argument(
        "--from-step",
        type=str,
        default=None,
        choices=PIPELINE_STEPS,
        help="First step to run (inclusive)",
    )
    parser.add_argument(
        "--to-step",
        type=str,
        default=None,
        choices=PIPELINE_STEPS,
        help="Last step to run (inclusive)",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        choices=PIPELINE_STEPS,
        help="Run only these steps",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps whose primary output files already exist",
    )
    return parser.parse_args()


def resolve_steps(
    args: argparse.Namespace,
    config_steps: list[str],
) -> list[str]:
    if args.only:
        return list(args.only)

    steps = list(config_steps)
    if args.from_step:
        start = steps.index(args.from_step)
        steps = steps[start:]
    if args.to_step:
        end = steps.index(args.to_step)
        steps = steps[: end + 1]
    return steps


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.run_full_pipeline")
    paths = ProjectPaths.from_root(_ROOT)
    config_path = paths.root / args.config

    config = load_config(config_path)
    pipeline_cfg = config.setdefault("pipeline", {})
    if args.skip_existing:
        pipeline_cfg["skip_if_outputs_exist"] = True

    config_steps = pipeline_cfg.get("steps", list(PIPELINE_STEPS))
    steps = resolve_steps(args, config_steps)

    logger.info("Config: %s", config_path)
    logger.info("Steps:  %s", " → ".join(steps))

    runner = FullPipelineRunner(config, paths)
    code = runner.run(steps)

    if code == 0:
        logger.info("Pipeline finished successfully.")
        print("Research evidence pipeline completed successfully.")
    else:
        logger.error("Pipeline exited with code %d.", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
