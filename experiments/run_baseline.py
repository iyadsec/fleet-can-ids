#!/usr/bin/env python3
"""Baseline experiment entry point (data + graph scaffold; no models yet)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as: python experiments/run_baseline.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.graph import build_fleet_graph
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fleet-Aware CAN-IDS baseline scaffold")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (relative to project root)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.run_baseline")
    config = load_config(args.config)
    paths = ProjectPaths.from_root(_ROOT)

    logger.info("Project root: %s", paths.root)
    logger.info("Experiment: %s", config.get("experiment", {}).get("name", "unnamed"))
    logger.info("Models are not implemented yet — graph scaffold only.")

    # Empty trace list until loaders are implemented
    graph = build_fleet_graph([], config)
    logger.info("Fleet graph: %d nodes, %d edges", graph.n_nodes, graph.n_edges)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
