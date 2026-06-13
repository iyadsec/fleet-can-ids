"""Hierarchical separation of local Isolation Forest vs fleet correlation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED, DECISION_ISOLATED

CONFIG_LABELS = {
    "C1": "Local-only IDS",
    "C2": "Similarity-only fleet correlation",
    "C3": "GraphSAGE-based fleet correlation",
}

METHOD_TO_CONFIG: dict[str, str] = {
    "local_ids": "C1",
    "descriptor_clustering": "C2",
    "fcgnn": "C3",
    "standard_gnn": "S_supplementary",
}


@dataclass(frozen=True)
class LocalThresholds:
    weak: float = 0.55
    strong: float = 0.80


def classify_local_evidence(
    score: float,
    *,
    local_alert: int,
    weak_signal: int,
    thresholds: LocalThresholds,
) -> str:
    if int(local_alert) == 1:
        return "strong"
    if int(weak_signal) == 1:
        return "weak"
    if score >= thresholds.strong:
        return "strong"
    if score >= thresholds.weak:
        return "weak"
    return "benign"


def map_fleet_decision(final_decision: str, *, method: str) -> str:
    if method == "local_ids":
        return "no_fleet_evidence"
    fd = str(final_decision or "")
    if fd == DECISION_COORDINATED:
        return "coordinated_campaign"
    if fd == DECISION_ISOLATED:
        return "isolated_incident"
    return "no_fleet_evidence"


def local_event_alert(row: pd.Series) -> int:
    """Strong local alert only (IF); weak evidence does not set this flag."""
    return int(row.get("local_alert", 0) or 0)


def local_if_event_positive(row: pd.Series) -> int:
    """Full IF suspicious output (strong or weak band) — not fleet-derived."""
    return int((int(row.get("local_alert", 0) or 0) == 1) or (int(row.get("weak_signal", 0) or 0) == 1))


def align_event_predictions(
    df: pd.DataFrame,
    *,
    method: str,
    thresholds: LocalThresholds | None = None,
) -> pd.DataFrame:
    """Map raw run predictions to hierarchical output schema."""
    thresholds = thresholds or LocalThresholds()
    out = df.copy()
    scores = pd.to_numeric(out["anomaly_score"], errors="coerce").fillna(0.0)

    out["local_anomaly_score"] = scores
    out["local_evidence_level"] = [
        classify_local_evidence(
            float(s),
            local_alert=int(out.iloc[i].get("local_alert", 0) or 0),
            weak_signal=int(out.iloc[i].get("weak_signal", 0) or 0),
            thresholds=thresholds,
        )
        for i, s in enumerate(scores)
    ]
    out["local_event_alert"] = out.apply(local_event_alert, axis=1).astype(int)

    if method == "local_ids":
        out["fleet_cluster_id"] = -1
        out["fleet_campaign_member"] = 0
        out["fleet_campaign_confidence"] = 0.0
        out["fleet_decision"] = "no_fleet_evidence"
    else:
        out["fleet_cluster_id"] = pd.to_numeric(out.get("cluster_id", -1), errors="coerce").fillna(-1).astype(int)
        out["fleet_campaign_member"] = (
            out.get("final_decision", pd.Series("", index=out.index)) == DECISION_COORDINATED
        ).astype(int)
        conf = pd.to_numeric(out.get("gnn_campaign_score", 0), errors="coerce").fillna(0.0)
        cohesion = pd.to_numeric(out.get("behavioral_cohesion", 0), errors="coerce").fillna(0.0)
        out["fleet_campaign_confidence"] = (0.6 * conf + 0.4 * cohesion).clip(0, 1)
        out["fleet_decision"] = out["final_decision"].map(
            lambda fd: map_fleet_decision(fd, method=method)
        )

    # Identity columns
    for col in ("event_id", "scenario_vehicle_id", "vehicle_token", "ground_truth_malicious"):
        if col not in out.columns:
            out[col] = np.nan

    out["framework_config"] = METHOD_TO_CONFIG.get(method, method)
    out["method"] = method
    return out[
        [
            "event_id",
            "scenario_vehicle_id",
            "vehicle_token",
            "local_anomaly_score",
            "local_evidence_level",
            "local_event_alert",
            "fleet_cluster_id",
            "fleet_campaign_member",
            "fleet_campaign_confidence",
            "fleet_decision",
            "ground_truth_malicious",
            "framework_config",
            "method",
        ]
        + [c for c in ("ground_truth_campaign_id", "scenario_role", "weak_signal", "local_alert") if c in out.columns]
    ]


def validate_local_not_overwritten(raw: pd.DataFrame, aligned: pd.DataFrame) -> list[str]:
    """Return validation errors if fleet layer altered local fields."""
    errors: list[str] = []
    if "local_alert" in raw.columns:
        if not (aligned["local_event_alert"].astype(int).values == raw["local_alert"].astype(int).values).all():
            errors.append("local_event_alert differs from source local_alert")
    return errors
