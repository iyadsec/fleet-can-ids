"""Load or fit fleet benign-training scaler into experiment config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.local_descriptor_normalisation import (
    FleetScalerProvenance,
    get_or_fit_global_scaler,
)
from src.utils.paths import resolve_project_root

DEFAULT_SCALER_CACHE = Path("new_experiments/metadata_correction/manifests/fleet_benign_scaler.json")


def ensure_fleet_scaler_in_config(
    config: dict[str, Any],
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
) -> FleetScalerProvenance:
    """Attach `_fleet_scaler_provenance` to config (idempotent)."""
    if config.get("_fleet_scaler_provenance"):
        return config["_fleet_scaler_provenance"]

    norm_cfg = config.get("fleet_normalisation", {})
    cache_rel = norm_cfg.get("scaler_cache", str(DEFAULT_SCALER_CACHE))
    cache_path = Path(cache_rel)
    if not cache_path.is_absolute():
        cache_path = resolve_project_root() / cache_path

    prov = get_or_fit_global_scaler(descriptors, manifest, cache_path)
    config["_fleet_scaler_provenance"] = prov
    return prov


def resolve_fleet_scaler_from_config(config: dict[str, Any]) -> FleetScalerProvenance:
    prov = config.get("_fleet_scaler_provenance")
    if prov is None:
        cache_path = resolve_project_root() / DEFAULT_SCALER_CACHE
        if cache_path.exists():
            from src.experiments.local_descriptor_normalisation import load_scaler_provenance

            return load_scaler_provenance(cache_path)
        raise ValueError(
            "Fleet scaler not initialised. Call ensure_fleet_scaler_in_config() before graph methods."
        )
    return prov
