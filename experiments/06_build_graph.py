#!/usr/bin/env python3
"""
Experiment 06: Build Fleet Graph (Publication Path)

Builds the FLEET-GUARD publication fleet anomaly graph using the
cross-vehicle constrained k-NN edge rule:

  - cosine similarity threshold τ
  - max_same_vehicle_neighbors (same-vehicle top-k)
  - max_cross_vehicle_neighbors (cross-vehicle top-k)

This is the methodology behind experimental-2026-06-23/
01_primary_ocslab_balanced/. Legacy radius graphs with a single
max_neighbors cap are not supported.

Usage:
    python experiments/06_build_graph.py
    python experiments/06_build_graph.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.fleet_graph_builder import (  # noqa: E402
    load_anomaly_descriptors,
    print_graph_statistics,
    save_fleet_graph,
    save_graph_tables,
)
from src.graph.scenario_graph import build_scenario_graph  # noqa: E402


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _stats_row_to_dict(stats_df: pd.DataFrame) -> dict:
    row = stats_df.iloc[0].to_dict() if len(stats_df) else {}
    return {
        "num_nodes": float(row.get("nodes", 0)),
        "num_edges": float(row.get("unique_undirected_edges", 0)),
        "average_degree": float(row.get("average_degree", 0.0)),
        "graph_density": float(row.get("graph_density", 0.0)),
        "num_cross_vehicle_edges": float(row.get("cross_vehicle_edges", 0)),
        "connected_components": float(row.get("connected_components", 0)),
        "similarity_threshold": float(row.get("similarity_threshold", 0.0)),
        "max_same_vehicle_neighbors": float(row.get("max_same_vehicle_neighbors", 0)),
        "max_cross_vehicle_neighbors": float(row.get("max_cross_vehicle_neighbors", 0)),
        "similarity_metric": row.get("similarity_metric", "cosine"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build publication fleet anomaly graph (constrained k-NN)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--descriptors",
        type=str,
        default=None,
        help="Path to anomaly descriptors CSV (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for graph artifacts (overrides config processed_dir)",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    paths = config.get("paths", {})
    artifacts = config.get("pipeline", {}).get("artifacts", {})
    graph_cfg = config.get("graph", {})
    seed = int(config.get("project", {}).get("seed", 42))

    descriptors_path = Path(
        args.descriptors
        or artifacts.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv")
    )
    processed_dir = Path(args.output_dir or paths.get("processed_dir", "data/processed"))
    outputs_dir = Path(paths.get("outputs_dir", "outputs"))
    metrics_dir = Path(paths.get("metrics_dir", "outputs/metrics"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pt_out = Path(artifacts.get("fleet_graph", str(processed_dir / "fleet_graph.pt")))
    graphml_out = Path(
        artifacts.get("fleet_graph_graphml", str(outputs_dir / "fleet_graph.graphml"))
    )
    nodes_out = Path(artifacts.get("fleet_nodes", str(processed_dir / "fleet_nodes.csv")))
    edges_out = Path(artifacts.get("fleet_edges", str(processed_dir / "fleet_edges.csv")))
    stats_out = Path(
        artifacts.get("graph_statistics", str(metrics_dir / "graph_statistics.csv"))
    )

    print("=" * 70)
    print("EXPERIMENT 06: Build Fleet Graph (Publication Constrained k-NN)")
    print("=" * 70)

    if not descriptors_path.exists():
        print(f"ERROR: Descriptors not found: {descriptors_path}")
        print("Run experiment 05 first to generate anomaly descriptors.")
        sys.exit(1)

    print(f"\n[1/4] Loading anomaly descriptors from {descriptors_path}...")
    feat_path = Path(artifacts.get("window_features", "data/processed/window_features.csv"))
    descriptors = load_anomaly_descriptors(
        descriptors_path,
        features_path=feat_path if feat_path.exists() else None,
    )
    print(f"  Loaded {len(descriptors)} descriptors")

    print("\n[2/4] Building publication constrained k-NN graph...")
    print(f"  similarity_threshold: {graph_cfg.get('similarity_threshold', 0.95)}")
    print(f"  max_same_vehicle_neighbors: {graph_cfg.get('max_same_vehicle_neighbors', 2)}")
    print(f"  max_cross_vehicle_neighbors: {graph_cfg.get('max_cross_vehicle_neighbors', 5)}")

    result = build_scenario_graph(
        descriptors,
        config=config,
        seed=seed,
        build_pyg=True,
    )
    stats = _stats_row_to_dict(result.stats)

    print("\n[3/4] Saving graph artifacts...")
    save_fleet_graph(
        result.graph,
        result.pyg_data,
        stats,
        pt_path=pt_out,
        graphml_path=graphml_out,
    )
    save_graph_tables(result.graph, nodes_path=nodes_out, edges_path=edges_out)
    pd.DataFrame(
        [
            {
                "num_nodes": int(stats["num_nodes"]),
                "num_edges": int(stats["num_edges"]),
                "similarity_threshold": float(stats["similarity_threshold"]),
                "max_same_vehicle_neighbors": int(stats["max_same_vehicle_neighbors"]),
                "max_cross_vehicle_neighbors": int(stats["max_cross_vehicle_neighbors"]),
                "num_cross_vehicle_edges": int(stats["num_cross_vehicle_edges"]),
                "num_same_vehicle_edges": int(
                    result.stats.iloc[0].get("same_vehicle_edges", 0)
                )
                if len(result.stats)
                else 0,
                "graph_density": float(stats["graph_density"]),
                "average_degree": float(stats["average_degree"]),
                "connected_components": int(stats["connected_components"]),
            }
        ]
    ).to_csv(stats_out, index=False)
    print_graph_statistics(stats)

    print("\n[4/4] Graph build complete.")
    print(f"  Nodes: {int(stats['num_nodes'])}")
    print(f"  Edges: {int(stats['num_edges'])}")
    print(f"  Same-vehicle edges: {int(result.stats.iloc[0].get('same_vehicle_edges', 0))}")
    print(f"  Cross-vehicle edges: {int(stats['num_cross_vehicle_edges'])}")
    print(f"  PyG: {pt_out}")
    print(f"  GraphML: {graphml_out}")
    print("\n" + "=" * 70)
    print("EXPERIMENT 06 COMPLETE")
    print("=" * 70)
    print("\nNext step: Run experiment 07 to train GraphSAGE")


if __name__ == "__main__":
    main()
