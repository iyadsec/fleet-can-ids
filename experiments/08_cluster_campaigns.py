#!/usr/bin/env python3
"""Cluster GNN embeddings to find suspicious multi-vehicle attack campaigns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.campaign_clustering import (
    load_embedding_table,
    plot_cluster_summaries,
    plot_tsne_clusters,
    print_cluster_report,
    run_campaign_clustering,
    save_campaign_clusters,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster GNN embeddings for campaign detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/processed/node_embeddings.csv",
        help="GNN embedding CSV",
    )
    parser.add_argument(
        "--descriptors",
        type=str,
        default="data/processed/anomaly_descriptors.csv",
        help="Metadata / behavioural fallback",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/window_features.csv",
        help="Window features for fast behavioural fallback (no timing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/fleet_cluster_results.csv",
    )
    parser.add_argument("--kmeans-k", type=int, default=12)
    parser.add_argument("--dbscan-eps", type=float, default=1.2)
    parser.add_argument("--dbscan-min-samples", type=int, default=10)
    parser.add_argument("--dbscan-pca-components", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument("--min-vehicles", type=int, default=2)
    parser.add_argument("--method", type=str, default=None, choices=["dbscan", "kmeans", "both"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Max samples for clustering fit (labels projected to all nodes)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.08_cluster_campaigns")
    paths = ProjectPaths.from_root(_ROOT)

    seed = args.seed if args.seed is not None else 42
    cluster_cfg: dict = {}
    if args.config:
        try:
            cfg = load_config(paths.root / args.config)
            seed = int(cfg.get("project", {}).get("seed", seed))
            cluster_cfg = cfg.get("clustering", {})
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    emb_path = paths.root / args.embeddings
    desc_path = paths.root / args.descriptors
    feat_path = paths.root / args.features
    out_path = paths.root / args.output
    figures_dir = paths.figures_dir

    k = args.kmeans_k if args.kmeans_k else int(cluster_cfg.get("kmeans_clusters", 12))
    eps = float(cluster_cfg.get("dbscan_eps", args.dbscan_eps))
    min_samples = int(cluster_cfg.get("dbscan_min_samples", args.dbscan_min_samples))
    pca_comp = int(cluster_cfg.get("dbscan_pca_components", args.dbscan_pca_components))
    sim_thr = args.similarity_threshold or float(cluster_cfg.get("similarity_threshold", 0.85))
    min_veh = args.min_vehicles or int(cluster_cfg.get("min_vehicles", 2))
    method = args.method or str(cluster_cfg.get("method", cluster_cfg.get("clustering_method", "dbscan"))).lower()

    logger.info("Embeddings: %s", emb_path)
    logger.info("Descriptors: %s", desc_path)

    X, meta = load_embedding_table(emb_path, desc_path, features_path=feat_path)
    max_samples = args.max_samples
    if cluster_cfg.get("max_clustering_samples"):
        max_samples = int(cluster_cfg["max_clustering_samples"])

    assignments = run_campaign_clustering(
        X,
        meta,
        similarity_threshold=sim_thr,
        min_vehicles=min_veh,
        kmeans_clusters=k,
        dbscan_eps=eps,
        dbscan_min_samples=min_samples,
        dbscan_pca_components=pca_comp,
        random_state=seed,
        max_clustering_samples=max_samples,
        method=method,
    )

    save_campaign_clusters(assignments, out_path)

    for algo in sorted(assignments["algorithm"].unique()):
        sub = assignments[assignments["algorithm"] == algo]
        plot_tsne_clusters(
            X,
            sub,
            meta,
            figures_dir / f"campaign_tsne_{algo}.png",
            algorithm=algo,
            seed=seed,
        )

    plot_cluster_summaries(assignments, figures_dir / "campaign_cluster_summary.png")
    print_cluster_report(assignments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
