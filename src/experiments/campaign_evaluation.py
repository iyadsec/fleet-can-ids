"""Event-, vehicle-, and campaign-level metrics for scenario experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_event_metrics(event_predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = event_predictions["ground_truth_malicious"].astype(int).to_numpy()
    y_pred = event_predictions["predicted_malicious"].astype(int).to_numpy()
    cm = compute_confusion(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = _safe_div(cm["fp"], cm["fp"] + cm["tn"])
    fnr = _safe_div(cm["fn"], cm["fn"] + cm["tp"])

    scores = event_predictions["anomaly_score"].astype(float).to_numpy()
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
            pr_auc = float(average_precision_score(y_true, scores))
        except ValueError:
            pass

    weak_gt = (
        (event_predictions.get("weak_signal", pd.Series(0, index=event_predictions.index)) == 1)
        & (event_predictions["ground_truth_malicious"] == 1)
    )
    weak_recovered = int(
        (
            weak_gt
            & (
                (event_predictions["predicted_malicious"] == 1)
                | (event_predictions["final_decision"] == DECISION_COORDINATED)
            )
        ).sum()
    )
    weak_fp = int(
        (
            (event_predictions.get("weak_signal", 0) == 1)
            & (event_predictions["ground_truth_malicious"] == 0)
            & (event_predictions["predicted_malicious"] == 1)
        ).sum()
    )

    return {
        **cm,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "weak_malicious_recovered": weak_recovered,
        "weak_benign_promoted": weak_fp,
    }


def compute_vehicle_metrics(vehicle_predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = vehicle_predictions["ground_truth_attacked"].astype(int).to_numpy()
    y_pred = vehicle_predictions["predicted_attacked"].astype(int).to_numpy()
    cm = compute_confusion(y_true, y_pred)
    return {
        **cm,
        "vehicle_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "vehicle_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "vehicle_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "attacked_vehicles_recovered": cm["tp"],
        "benign_vehicles_incorrectly_included": cm["fp"],
    }


def compute_campaign_metrics(
    event_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    spec_expects_campaign: bool,
) -> dict[str, Any]:
    gt_campaigns = membership.loc[
        membership["ground_truth_campaign_id"].astype(str).str.len() > 0, "ground_truth_campaign_id"
    ].unique()
    n_gt = len(set(gt_campaigns))
    qualifying = (
        cluster_df[cluster_df["is_qualifying_campaign_cluster"]]
        if not cluster_df.empty and "is_qualifying_campaign_cluster" in cluster_df.columns
        else pd.DataFrame()
    )
    n_detected = len(qualifying)
    coordinated_events = int((event_predictions["final_decision"] == DECISION_COORDINATED).sum())

    if spec_expects_campaign and n_gt > 0:
        detected = int(n_detected >= 1 and coordinated_events > 0)
        recall = float(detected)
        precision = _safe_div(
            int(n_detected > 0 and n_detected <= max(n_gt, 1)),
            max(n_detected, 1),
        )
    elif not spec_expects_campaign:
        detected = int(n_detected > 0)
        recall = 1.0 - float(detected)  # success = no false campaign
        precision = 1.0 if n_detected == 0 else 0.0
    else:
        detected = 0
        recall = precision = 0.0

    f1 = _safe_div(2 * precision * recall, precision + recall)
    false_campaign_rate = _safe_div(n_detected, max(n_detected + int(not spec_expects_campaign), 1))

    incorrect_merge = 0
    if membership["scenario_id"].iloc[0] == "S2" and n_detected == 1 and n_gt > 1:
        incorrect_merge = 1

    return {
        "campaign_detection_rate": float(detected) if spec_expects_campaign else float(1 - detected),
        "campaign_precision": float(precision),
        "campaign_recall": float(recall),
        "campaign_f1": float(f1),
        "false_campaign_alert_rate": float(false_campaign_rate),
        "n_detected_campaign_clusters": n_detected,
        "n_ground_truth_campaigns": n_gt,
        "incorrect_campaign_merging": incorrect_merge,
        "coordinated_events": coordinated_events,
    }


def aggregate_run_metrics(
    *,
    method: str,
    seed: int,
    scenario_key: str,
    campaign_size: int,
    coordination_strength: float,
    event_predictions: pd.DataFrame,
    vehicle_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    expect_campaign: bool,
    runtime: dict[str, float],
) -> dict[str, Any]:
    ev = compute_event_metrics(event_predictions)
    veh = compute_vehicle_metrics(vehicle_predictions)
    camp = compute_campaign_metrics(event_predictions, membership, cluster_df, expect_campaign)
    return {
        "method": method,
        "seed": seed,
        "scenario_key": scenario_key,
        "campaign_size": campaign_size,
        "coordination_strength": coordination_strength,
        **ev,
        **veh,
        **camp,
        **{f"runtime_{k}": v for k, v in runtime.items()},
    }
