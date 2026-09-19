"""Intrusion detection models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "generate_vehicle_anomaly_predictions",
    "generate_window_predictions",
    "load_window_predictions",
    "run_vehicle_level_training",
    "save_results",
    "save_vehicle_anomaly_predictions",
    "save_window_predictions",
    "train_all_models_for_vehicle",
    "GraphSAGEFleetCorrelator",
    "train_graphsage_fleet_correlation",
]


def __getattr__(name: str) -> Any:
    if name in {
        "generate_vehicle_anomaly_predictions",
        "generate_window_predictions",
        "load_window_predictions",
        "run_vehicle_level_training",
        "save_results",
        "save_vehicle_anomaly_predictions",
        "save_window_predictions",
        "train_all_models_for_vehicle",
    }:
        from src.models import vehicle_ids as _vehicle_ids

        return getattr(_vehicle_ids, name)
    if name in {"GraphSAGEFleetCorrelator", "train_graphsage_fleet_correlation"}:
        from src.models import gnn_models as _gnn

        return getattr(_gnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
