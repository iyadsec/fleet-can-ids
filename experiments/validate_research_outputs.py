#!/usr/bin/env python3
"""Validate research evidence outputs produced by the full pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


REQUIRED_FILES = [
    "data/processed/vehicle_anomaly_predictions.csv",
    "data/processed/anomaly_descriptors.csv",
    "data/processed/fleet_nodes.csv",
    "data/processed/fleet_edges.csv",
    "data/processed/fleet_graph.pt",
    "data/processed/node_embeddings.csv",
    "data/processed/fleet_cluster_results.csv",
    "outputs/metrics/raw_vs_descriptor_size.csv",
    "outputs/metrics/graph_statistics.csv",
    "outputs/metrics/final_detection_outcomes.csv",
    "outputs/metrics/fleet_value_summary.csv",
    "outputs/metrics/weak_signal_upgrade_summary.csv",
    "outputs/metrics/cross_vehicle_cluster_summary.csv",
]

EXPECTED_OUTCOMES = {
    "Isolated anomaly",
    "Weak isolated signal",
    "Fleet-level coordinated behavioural pattern",
}


def _failures_for_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing: {path}"]
    if path.stat().st_size == 0:
        return [f"empty: {path}"]
    return []


def main() -> int:
    root = _ROOT
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        failures.extend(_failures_for_file(root / rel))

    if failures:
        print("\n".join(failures))
        return 1

    predictions = pd.read_csv(root / "data/processed/vehicle_anomaly_predictions.csv")
    if "weak_signal" not in predictions.columns:
        failures.append("weak_signal missing in vehicle_anomaly_predictions.csv")

    outcomes = pd.read_csv(root / "outputs/metrics/final_detection_outcomes.csv")
    if "final_outcome" not in outcomes.columns:
        failures.append("final_outcome missing in final_detection_outcomes.csv")
    else:
        found = set(outcomes["final_outcome"].dropna().unique())
        if not found.intersection(EXPECTED_OUTCOMES):
            failures.append("final_outcome does not contain expected categories")
    if "was_upgraded_by_fleet" not in outcomes.columns:
        failures.append("was_upgraded_by_fleet missing in final_detection_outcomes.csv")

    clusters = pd.read_csv(root / "data/processed/fleet_cluster_results.csv")
    if "is_cross_vehicle_cluster" not in clusters.columns:
        failures.append("is_cross_vehicle_cluster missing in fleet_cluster_results.csv")
    elif clusters["num_unique_vehicles"].max() > 1 and not clusters["is_cross_vehicle_cluster"].any():
        failures.append("cross-vehicle clusters expected but none marked")

    metric_files = [
        "outputs/metrics/raw_vs_descriptor_size.csv",
        "outputs/metrics/graph_statistics.csv",
        "outputs/metrics/fleet_value_summary.csv",
        "outputs/metrics/weak_signal_upgrade_summary.csv",
        "outputs/metrics/cross_vehicle_cluster_summary.csv",
    ]
    for rel in metric_files:
        df = pd.read_csv(root / rel)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.isna().any().any():
            failures.append(f"NaN found in numeric metrics: {rel}")

    if failures:
        print("Research output validation failed:")
        for failure in failures:
            print(f" - {failure}")
        return 1

    print("Research output validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
