#!/usr/bin/env python3
"""
Fleet-aware coordinated CAN attack campaign detection evaluation.

Usage:
  python run_campaign_detection_evaluation.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.campaign_detection_experiment import (
    CampaignDetectionConfig,
    CampaignDetectionOutputs,
    run_campaign_detection_experiment,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_campaign_detection_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fleet campaign detection evaluation (IEEE-ready)")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    cd = config.get("campaign_detection", {})
    fg = parse_fleet_graph_similarity_settings(config)
    seed = int(get_nested(config, "project", "seed", default=42))

    fleet_graph_cfg = config.get("fleet_graph", {})
    cfg = CampaignDetectionConfig(
        top_k_same_vehicle=int(cd.get("top_k_same_vehicle", fleet_graph_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(cd.get("top_k_cross_vehicle", fleet_graph_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(cd.get("similarity_threshold", fleet_graph_cfg.get("similarity_threshold", 0.95))),
        similarity_feature_view=fg["similarity_feature_view"],
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        allowed_high_dominance_features=fg["allowed_high_dominance_features"],
        min_vehicles=int(cd.get("min_vehicles", 2)),
        min_cluster_size=int(cd.get("min_cluster_size", 10)),
        min_cohesion=float(cd.get("min_cohesion", 0.85)),
        min_dominant_attack_ratio=float(cd.get("min_dominant_attack_ratio", 0.60)),
        campaign_match_recall=float(cd.get("campaign_match_recall", 0.15)),
        campaign_match_min_nodes=int(cd.get("campaign_match_min_nodes", 20)),
        dbscan_eps=float(cd.get("dbscan_eps", config.get("clustering", {}).get("dbscan_eps", 1.2))),
        dbscan_min_samples=int(
            cd.get("dbscan_min_samples", config.get("clustering", {}).get("dbscan_min_samples", 10))
        ),
        dbscan_pca_components=int(
            cd.get("dbscan_pca_components", config.get("clustering", {}).get("dbscan_pca_components", 8))
        ),
        max_clustering_samples=int(
            cd.get("max_clustering_samples", config.get("clustering", {}).get("max_clustering_samples", 20000))
        ),
        max_graph_viz_nodes=int(cd.get("max_graph_viz_nodes", 800)),
        max_embedding_samples=int(cd.get("max_embedding_samples", 5000)),
        embedding_method=str(cd.get("embedding_method", "tsne")),  # type: ignore[arg-type]
        seed=seed,
    )

    descriptors_path = paths.root / artifacts.get(
        "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
    )
    features_path = paths.root / artifacts.get(
        "window_features", "data/processed/window_features.csv"
    )
    for p, name in [(descriptors_path, "anomaly_descriptors"), (features_path, "window_features")]:
        if not p.exists():
            logger.error("Missing %s: %s", name, p)
            return 1

    outputs = CampaignDetectionOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_campaign_detection_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for key, path in written.items():
        logger.info("%s -> %s", key, path)
    print("Campaign detection evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
