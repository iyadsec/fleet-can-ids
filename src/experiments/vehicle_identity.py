"""Opaque vehicle-instance tokens and graph identity resolution."""

from __future__ import annotations

import pandas as pd


def resolve_graph_vehicle_column(meta: pd.DataFrame) -> str:
    """
    Operational graph vehicle identifier for kNN constraints and PyG vehicle_id.

    Priority: vehicle_token → scenario_vehicle_id → vehicle_id → vehicle_model.
    ``vehicle_id`` / ``vehicle_model`` are accepted for CTT adapters that map
    dataset vehicle strings into the same operational role as OCSLab tokens.
    """
    for col in ("vehicle_token", "scenario_vehicle_id", "vehicle_id", "vehicle_model"):
        if col in meta.columns and meta[col].notna().any():
            return col
    raise ValueError(
        "Fleet graph requires a vehicle identity column "
        "(vehicle_token / scenario_vehicle_id / vehicle_id / vehicle_model)."
    )
