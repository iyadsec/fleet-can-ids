"""Benign-training scaler for fleet-layer descriptor views (no vehicle_model grouping)."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.data_splits import is_benign_attack_type
from src.graph.fleet_similarity_features import (
    BEHAVIOR_GRAPH_CANDIDATE_COLUMNS,
    build_behavior_view_descriptors,
)

DEFAULT_FLEET_NORM_FEATURES: tuple[str, ...] = BEHAVIOR_GRAPH_CANDIDATE_COLUMNS + ("payload_entropy",)


@dataclass
class FleetScalerProvenance:
    scaler_id: str
    training_split: str
    source_vehicle_token: str
    fitted_feature_names: list[str]
    fit_row_count: int
    means: dict[str, float]
    stds: dict[str, float]
    attack_labels_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FleetScalerProvenance:
        return cls(**data)


def _benign_training_rows(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    split: str = "train",
) -> pd.DataFrame:
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns]
    desc = descriptors.merge(
        manifest[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )
    benign = desc["attack_type"].map(is_benign_attack_type)
    return desc.loc[(desc["split"] == split) & benign].copy()


def fit_benign_fleet_scaler(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | None = None,
    training_split: str = "train",
    source_vehicle_token: str = "global",
) -> FleetScalerProvenance:
    """
    Fit z-score statistics on benign training rows only.

    Never uses validation/test rows or attack labels for fitting.
    """
    features = list(feature_names or DEFAULT_FLEET_NORM_FEATURES)
    train = _benign_training_rows(descriptors, manifest, split=training_split)
    if train.empty:
        raise ValueError("No benign training rows available for fleet scaler fit")

    view = build_behavior_view_descriptors(train)
    if "payload_entropy" in features and "payload_entropy" not in view.columns:
        from src.evaluation.final_gnn_fleet_decision_experiment import compute_payload_entropy

        view["payload_entropy"] = compute_payload_entropy(view)

    cols = [c for c in features if c in view.columns]
    if not cols:
        raise ValueError("No fleet normalisation features found in benign training data")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for col in cols:
        s = view[col].astype(np.float64)
        means[col] = float(s.mean())
        stds[col] = float(s.std(ddof=0) if len(s) > 1 else 1.0)
        if stds[col] < 1e-9:
            stds[col] = 1.0

    return FleetScalerProvenance(
        scaler_id=f"fleet_benign_{uuid.uuid4().hex[:12]}",
        training_split=training_split,
        source_vehicle_token=source_vehicle_token,
        fitted_feature_names=cols,
        fit_row_count=int(len(view)),
        means=means,
        stds=stds,
        attack_labels_used=False,
    )


def apply_fleet_scaler(
    descriptors: pd.DataFrame,
    provenance: FleetScalerProvenance,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply pre-fitted benign-training z-scores. Does not use vehicle_model."""
    out = build_behavior_view_descriptors(descriptors.copy())
    if "payload_entropy" in (feature_names or provenance.fitted_feature_names):
        if "payload_entropy" not in out.columns:
            from src.evaluation.final_gnn_fleet_decision_experiment import compute_payload_entropy

            out["payload_entropy"] = compute_payload_entropy(out)

    cols = feature_names or tuple(provenance.fitted_feature_names)
    for col in cols:
        if col not in out.columns:
            continue
        mean = provenance.means.get(col, 0.0)
        std = provenance.stds.get(col, 1.0)
        out[col] = (out[col].astype(np.float64) - mean) / (std + 1e-9)
    return out


def save_scaler_provenance(provenance: FleetScalerProvenance, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")
    return path


def load_scaler_provenance(path: Path) -> FleetScalerProvenance:
    return FleetScalerProvenance.from_dict(json.loads(path.read_text(encoding="utf-8")))


def get_or_fit_global_scaler(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    cache_path: Path,
) -> FleetScalerProvenance:
    if cache_path.exists():
        return load_scaler_provenance(cache_path)
    prov = fit_benign_fleet_scaler(descriptors, manifest)
    save_scaler_provenance(prov, cache_path)
    return prov
