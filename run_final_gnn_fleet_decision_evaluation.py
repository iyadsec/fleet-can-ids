#!/usr/bin/env python3
"""
GNN-based fleet correlation with final isolated vs coordinated attack decisions.

Usage:
  python run_final_gnn_fleet_decision_evaluation.py --config configs/fleet_ids.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.final_gnn_fleet_decision_experiment import (
    FinalGnnFleetConfig,
    FinalGnnFleetOutputs,
    run_final_gnn_fleet_decision_experiment,
)
from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("run_final_gnn_fleet_decision_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final GNN fleet isolated vs coordinated decision evaluation")
    p.add_argument("--config", type=str, default="configs/fleet_ids.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    fg = parse_fleet_graph_similarity_settings(config)
    fleet_graph_cfg = config.get("fleet_graph", {})
    gnn_cfg = config.get("gnn", {})
    fd = config.get("final_gnn_fleet_decision", {})
    seed = int(get_nested(config, "project", "seed", default=42))

    ckpt = fd.get("checkpoint_path", "outputs/models/final_graphsage_fleet.pt")
    cfg = FinalGnnFleetConfig(
        top_k_same_vehicle=int(fd.get("top_k_same_vehicle", fleet_graph_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(fd.get("top_k_cross_vehicle", fleet_graph_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(fd.get("similarity_threshold", fleet_graph_cfg.get("similarity_threshold", 0.95))),
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        gnn_hidden_channels=int(fd.get("gnn_hidden_channels", gnn_cfg.get("hidden_channels", 64))),
        gnn_embedding_dim=int(fd.get("gnn_embedding_dim", gnn_cfg.get("embedding_dim", 32))),
        gnn_epochs=int(fd.get("gnn_epochs", gnn_cfg.get("epochs", 30))),
        gnn_learning_rate=float(fd.get("gnn_learning_rate", gnn_cfg.get("learning_rate", 0.01))),
        gnn_weight_decay=float(fd.get("gnn_weight_decay", gnn_cfg.get("weight_decay", 5e-4))),
        gnn_train_ratio=float(fd.get("gnn_train_ratio", gnn_cfg.get("train_ratio", 0.7))),
        gnn_val_ratio=float(fd.get("gnn_val_ratio", gnn_cfg.get("val_ratio", 0.15))),
        campaign_score_threshold=float(fd.get("campaign_score_threshold", 0.55)),
        min_cluster_size=int(fd.get("min_cluster_size", 10)),
        min_vehicles=int(fd.get("min_vehicles", 2)),
        min_dominant_attack_ratio=float(fd.get("min_dominant_attack_ratio", 0.80)),
        dbscan_eps=float(fd.get("dbscan_eps", config.get("clustering", {}).get("dbscan_eps", 1.2))),
        dbscan_min_samples=int(fd.get("dbscan_min_samples", config.get("clustering", {}).get("dbscan_min_samples", 10))),
        dbscan_pca_components=int(fd.get("dbscan_pca_components", config.get("clustering", {}).get("dbscan_pca_components", 8))),
        max_clustering_samples=int(fd.get("max_clustering_samples", 20000)),
        max_graph_viz_nodes=int(fd.get("max_graph_viz_nodes", 800)),
        max_embedding_samples=int(fd.get("max_embedding_samples", 5000)),
        embedding_method=str(fd.get("embedding_method", "tsne")),  # type: ignore[arg-type]
        checkpoint_path=paths.root / ckpt,
        retrain_gnn=bool(fd.get("retrain_gnn", True)),
        seed=seed,
    )

    descriptors_path = paths.root / artifacts.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv")
    features_path = paths.root / artifacts.get("window_features", "data/processed/window_features.csv")
    for p, name in [(descriptors_path, "anomaly_descriptors"), (features_path, "window_features")]:
        if not p.exists():
            logger.error("Missing %s: %s", name, p)
            return 1

    outputs = FinalGnnFleetOutputs(
        results_dir=paths.root / str(pub.get("results_dir", "results")),
        tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
        figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
    )

    written = run_final_gnn_fleet_decision_experiment(
        descriptors_path=descriptors_path,
        features_path=features_path,
        outputs=outputs,
        cfg=cfg,
    )
    for key, path in written.items():
        logger.info("%s -> %s", key, path)
    print("Final GNN fleet decision evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
