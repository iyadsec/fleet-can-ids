#!/usr/bin/env python3
"""Train a GCN on the fleet anomaly graph and export node embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.gnn_models import train_gnn_from_graph_file
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN on fleet anomaly graph")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--graph",
        type=str,
        default="data/processed/fleet_graph.pt",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="outputs/embeddings/gcn_node_embeddings.pt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="outputs/metrics/gnn_training_metrics.json",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.07_train_gnn")
    paths = ProjectPaths.from_root(_ROOT)

    seed = args.seed if args.seed is not None else 42
    gnn_cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            seed = int(cfg.get("project", {}).get("seed", seed))
            gnn_cfg = cfg.get("gnn", {})
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    graph_path = paths.root / args.graph
    emb_path = paths.root / args.embeddings
    ckpt_dir = paths.root / args.checkpoint_dir
    metrics_path = paths.root / args.metrics_output

    if not graph_path.exists():
        logger.error("Fleet graph not found: %s (run 06_build_graph.py)", graph_path)
        return 1

    logger.info("Graph:       %s", graph_path)
    logger.info("Embeddings:  %s", emb_path)

    metrics = train_gnn_from_graph_file(
        graph_path,
        emb_path,
        ckpt_dir,
        config=gnn_cfg,
        seed=seed,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote training metrics to %s", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
