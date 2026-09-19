"""Benign-training scaler for fleet-layer descriptor views (no vehicle grouping)."""

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
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _ensure_payload_entropy(view: pd.DataFrame) -> pd.DataFrame:
    if "payload_entropy" in view.columns:
        return view
    from src.evaluation.publication_fleet_core import compute_payload_entropy

    out = view.copy()
    out["payload_entropy"] = compute_payload_entropy(out)
    return out


def fit_benign_fleet_scaler_from_rows(
    benign_rows: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | None = None,
    training_split: str = "train",
    source_vehicle_token: str = "global",
) -> FleetScalerProvenance:
    """Fit z-score statistics on provided benign training rows only."""
    features = list(feature_names or DEFAULT_FLEET_NORM_FEATURES)
    if benign_rows.empty:
        raise ValueError("No benign training rows available for fleet scaler fit")

    view = build_behavior_view_descriptors(benign_rows)
    if "payload_entropy" in features:
        view = _ensure_payload_entropy(view)

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


def fit_benign_fleet_scaler(
    descriptors: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] | None = None,
    training_split: str = "train",
    source_vehicle_token: str = "global",
    split_column: str = "subset_name",
    train_prefix: str = "train",
) -> FleetScalerProvenance:
    """
    Fit z-score statistics on benign training rows only.

    Never uses validation/test rows or attack labels for the scaler objective.
    Benign membership is used only to select fitting rows (same as publication).
    """
    work = descriptors.copy()
    if split_column in work.columns:
        train_mask = work[split_column].astype(str).str.startswith(train_prefix)
    else:
        train_mask = pd.Series(True, index=work.index)
    if "attack_type" in work.columns:
        benign_mask = work["attack_type"].map(is_benign_attack_type)
    elif "label" in work.columns:
        benign_mask = work["label"].astype(int) == 0
    else:
        raise ValueError("Descriptors need attack_type or label to select benign train rows")
    train = work.loc[train_mask & benign_mask].copy()
    return fit_benign_fleet_scaler_from_rows(
        train,
        feature_names=feature_names,
        training_split=training_split,
        source_vehicle_token=source_vehicle_token,
    )


def apply_fleet_scaler(
    descriptors: pd.DataFrame,
    provenance: FleetScalerProvenance,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Apply pre-fitted benign-training z-scores. Does not use vehicle identity."""
    out = build_behavior_view_descriptors(descriptors.copy())
    cols = feature_names or tuple(provenance.fitted_feature_names)
    if "payload_entropy" in cols:
        out = _ensure_payload_entropy(out)
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
