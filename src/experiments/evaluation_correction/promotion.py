"""Corrected event promotion and three-label decision schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
from src.experiments.data_splits import is_benign_attack_type

AttackStrength = Literal["strong", "weak"]
LocalEvidence = Literal["benign", "weak_local_anomaly", "strong_local_anomaly"]


@dataclass(frozen=True)
class PromotionConfig:
    weak_threshold: float = 0.55
    strong_threshold: float = 0.80
    min_vehicles_in_cluster: int = 2
    min_behavioral_cohesion: float = 0.85
    min_cluster_size: int = 10
    promotion_confidence_threshold: float = 0.55
    min_malicious_support_fraction: float = 0.10


def classify_local_evidence(
    anomaly_score: float,
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> LocalEvidence:
    if anomaly_score >= strong_threshold:
        return "strong_local_anomaly"
    if anomaly_score >= weak_threshold:
        return "weak_local_anomaly"
    return "benign"


def classify_local_evidence_row(
    row: pd.Series,
    *,
    weak_threshold: float,
    strong_threshold: float,
) -> LocalEvidence:
    """Prefer pipeline flags (local_alert / weak_signal) over raw score bands."""
    if int(row.get("local_alert", 0) or 0) == 1:
        return "strong_local_anomaly"
    if int(row.get("weak_signal", 0) or 0) == 1:
        return "weak_local_anomaly"
    return classify_local_evidence(
        float(row.get("anomaly_score", 0.0)),
        weak_threshold=weak_threshold,
        strong_threshold=strong_threshold,
    )


def compute_fleet_event_confidence(row: pd.Series, cfg: PromotionConfig) -> float:
    """Confidence for weak-event promotion (0–1)."""
    score = float(row.get("anomaly_score", 0.0))
    score_norm = (score - cfg.weak_threshold) / max(cfg.strong_threshold - cfg.weak_threshold, 1e-9)
    score_norm = float(np.clip(score_norm, 0.0, 1.0))
    cohesion = float(row.get("behavioral_cohesion", 0.0) or 0.0)
    cohesion_norm = float(np.clip(cohesion / max(cfg.min_behavioral_cohesion, 1e-9), 0.0, 1.0))
    vehicles = int(row.get("vehicles_in_cluster", 0) or 0)
    vehicle_norm = float(np.clip(vehicles / max(cfg.min_vehicles_in_cluster, 1), 0.0, 1.0))
    gnn = float(row.get("gnn_campaign_score", 0.0) or 0.0)
    gnn_norm = float(np.clip(gnn, 0.0, 1.0))
    return float(0.35 * score_norm + 0.30 * cohesion_norm + 0.20 * vehicle_norm + 0.15 * gnn_norm)


def _cluster_is_qualifying(row: pd.Series, qualifying_ids: set[int]) -> bool:
    cid = int(row.get("cluster_id", -1))
    return cid in qualifying_ids


def _weak_promotion_allowed(
    row: pd.Series,
    *,
    cfg: PromotionConfig,
    qualifying_ids: set[int],
    cluster_malicious_fraction: dict[int, float],
) -> bool:
    if classify_local_evidence_row(
        row,
        weak_threshold=cfg.weak_threshold,
        strong_threshold=cfg.strong_threshold,
    ) != "weak_local_anomaly":
        return False
    if not _cluster_is_qualifying(row, qualifying_ids):
        return False
    if int(row.get("vehicles_in_cluster", 0) or 0) < cfg.min_vehicles_in_cluster:
        return False
    if float(row.get("behavioral_cohesion", 0.0) or 0.0) < cfg.min_behavioral_cohesion:
        return False
    cid = int(row.get("cluster_id", -1))
    if cluster_malicious_fraction.get(cid, 0.0) < cfg.min_malicious_support_fraction:
        return False
    conf = compute_fleet_event_confidence(row, cfg)
    return conf >= cfg.promotion_confidence_threshold


def apply_corrected_event_decisions(
    event_df: pd.DataFrame,
    *,
    attack_strength: AttackStrength,
    method: str,
    cfg: PromotionConfig | None = None,
    qualifying_cluster_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Apply three-label schema:
    - local_evidence_level
    - predicted_malicious_event (binary)
    - predicted_campaign_membership (fleet campaign)
    """
    cfg = cfg or PromotionConfig()
    out = event_df.copy()
    scores = out["anomaly_score"].astype(float)
    out["local_evidence_level"] = [
        classify_local_evidence_row(r, weak_threshold=cfg.weak_threshold, strong_threshold=cfg.strong_threshold)
        for _, r in out.iterrows()
    ]

    if qualifying_cluster_ids is None:
        qualifying_cluster_ids = set()
        if "final_decision" in out.columns:
            qualifying_cluster_ids = set(
                out.loc[out["final_decision"] == DECISION_COORDINATED, "cluster_id"].astype(int).unique()
            )

    cluster_mal_frac: dict[int, float] = {}
    if "cluster_id" in out.columns:
        for cid, grp in out.groupby("cluster_id"):
            if int(cid) < 0:
                continue
            strong_n = int((grp["local_evidence_level"] == "strong_local_anomaly").sum())
            cluster_mal_frac[int(cid)] = strong_n / max(len(grp), 1)

    out["predicted_campaign_membership"] = (
        out["final_decision"] == DECISION_COORDINATED
        if "final_decision" in out.columns
        else False
    ).astype(int)

    out["fleet_event_confidence"] = out.apply(lambda r: compute_fleet_event_confidence(r, cfg), axis=1)

    predicted: list[int] = []
    promoted_weak: list[int] = []
    for _, row in out.iterrows():
        pred = 0
        promoted = 0
        atk = str(row.get("attack_type", ""))
        evidence = row["local_evidence_level"]

        if is_benign_attack_type(atk):
            predicted.append(0)
            promoted_weak.append(0)
            continue

        if method == "local_ids":
            if attack_strength == "strong" and evidence == "strong_local_anomaly":
                pred = 1
            elif attack_strength == "weak" and evidence == "weak_local_anomaly":
                pred = 1
        else:
            if evidence == "strong_local_anomaly":
                pred = 1
            elif evidence == "weak_local_anomaly" and _weak_promotion_allowed(
                row, cfg=cfg, qualifying_ids=qualifying_cluster_ids, cluster_malicious_fraction=cluster_mal_frac
            ):
                pred = 1
                promoted = 1

        predicted.append(pred)
        promoted_weak.append(promoted)

    out["weak_malicious_promoted"] = promoted_weak
    out["predicted_malicious"] = predicted
    out["benign_incorrectly_promoted"] = (
        (out["ground_truth_malicious"] == 0) & (out["predicted_malicious"] == 1)
    ).astype(int)
    return out


def tune_promotion_threshold_validation(
    validation_frames: list[pd.DataFrame],
    *,
    attack_strength: AttackStrength,
    cfg: PromotionConfig,
    thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65),
) -> float:
    """Select promotion confidence threshold on validation split only."""
    best_t = cfg.promotion_confidence_threshold
    best_f1 = -1.0
    for t in thresholds:
        trial_cfg = PromotionConfig(**{**cfg.__dict__, "promotion_confidence_threshold": t})
        f1s: list[float] = []
        for raw in validation_frames:
            corrected = apply_corrected_event_decisions(
                raw, attack_strength=attack_strength, method="descriptor_clustering", cfg=trial_cfg
            )
            y_true = corrected["ground_truth_malicious"].astype(int).to_numpy()
            y_pred = corrected["predicted_malicious"].astype(int).to_numpy()
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_t = t
    return best_t
