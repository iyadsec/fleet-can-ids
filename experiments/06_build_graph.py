#!/usr/bin/env python3
"""Build fleet-level anomaly similarity graph from descriptors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.graph.fleet_graph_builder import (
    build_fleet_anomaly_graph,
    load_anomaly_descriptors,
    print_graph_statistics,
    save_fleet_graph,
    save_graph_tables,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fleet anomaly graph (behavioural similarity edges only)"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--descriptors",
        type=str,
        default="data/processed/anomaly_descriptors.csv",
    )
    parser.add_argument(
        "--pt-output",
        type=str,
        default="data/processed/fleet_graph.pt",
    )
    parser.add_argument(
        "--nodes-output",
        type=str,
        default="data/processed/fleet_nodes.csv",
    )
    parser.add_argument(
        "--edges-output",
        type=str,
        default="data/processed/fleet_edges.csv",
    )
    parser.add_argument(
        "--graphml-output",
        type=str,
        default="outputs/fleet_graph.graphml",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="outputs/metrics/fleet_graph_stats.json",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean"],
        default=None,
        help="Similarity metric (default: cosine from config)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum similarity for an edge",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Subsample nodes (for faster experiments)",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=None,
        help="Connect each node to at most this many nearest neighbours before thresholding",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.06_build_graph")
    paths = ProjectPaths.from_root(_ROOT)

    metric = args.metric or "cosine"
    threshold = args.threshold if args.threshold is not None else 0.85
    max_nodes = args.max_nodes
    max_neighbors = args.max_neighbors
    seed = args.seed if args.seed is not None else 42
    use_gt_labels = True
    cfg: dict = {}

    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            graph_cfg = cfg.get("graph", {})
            metric = args.metric or graph_cfg.get("similarity_metric", metric)
            if args.threshold is None:
                threshold = float(graph_cfg.get("similarity_threshold", threshold))
            if args.max_nodes is None and graph_cfg.get("max_nodes"):
                max_nodes = int(graph_cfg["max_nodes"])
            if args.max_neighbors is None and graph_cfg.get("max_neighbors"):
                max_neighbors = int(graph_cfg["max_neighbors"])
            seed = int(cfg.get("project", {}).get("seed", seed))
            use_gt_labels = bool(cfg.get("gnn", {}).get("use_ground_truth_labels", True))
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    descriptors_path = paths.root / args.descriptors
    pt_path = paths.root / args.pt_output
    nodes_path = paths.root / args.nodes_output
    edges_path = paths.root / args.edges_output
    graphml_path = paths.root / args.graphml_output
    stats_path = paths.root / args.stats_output

    if not descriptors_path.exists():
        logger.error("Descriptors not found: %s (run 05_generate_descriptors.py)", descriptors_path)
        return 1

    logger.info("Descriptors: %s", descriptors_path)
    logger.info("Metric:      %s, threshold: %.4f", metric, threshold)

    features_path = paths.root / "data/processed/window_features.csv"
    descriptors = load_anomaly_descriptors(
        descriptors_path, features_path=features_path if features_path.exists() else None
    )
    G, pyg_data, stats, _ = build_fleet_anomaly_graph(
        descriptors,
        metric=metric,  # type: ignore[arg-type]
        threshold=threshold,
        max_nodes=max_nodes,
        max_neighbors=max_neighbors,
        seed=seed,
        prefer_ground_truth_labels=use_gt_labels,
    )

    save_fleet_graph(G, pyg_data, stats, pt_path=pt_path, graphml_path=graphml_path)
    save_graph_tables(G, nodes_path=nodes_path, edges_path=edges_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote stats to %s", stats_path)

    print_graph_statistics(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
