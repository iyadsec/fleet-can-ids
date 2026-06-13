"""Distinct false-campaign error semantics (A–D) — separate from legacy aggregate rate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
from src.experiments.campaign_evaluation import compute_confusion, _safe_div
from src.experiments.experiment_pipeline import resolve_vehicle_id_column


def _n_ground_truth_campaigns(membership: pd.DataFrame) -> int:
    if membership.empty:
        return 0
    ids = membership.loc[
        membership["ground_truth_campaign_id"].astype(str).str.len() > 0, "ground_truth_campaign_id"
    ]
    return int(ids.nunique())


def _accepted_campaign_clusters(cluster_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_df.empty:
        return cluster_df
    if "campaign_accepted" in cluster_df.columns:
        return cluster_df[cluster_df["campaign_accepted"] == True]  # noqa: E712
    if "is_qualifying_campaign_cluster" in cluster_df.columns:
        return cluster_df[cluster_df["is_qualifying_campaign_cluster"]]
    return cluster_df


def compute_false_campaign_breakdown(
    event_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    scenario: str = "",
    expect_campaign: bool = True,
) -> dict[str, Any]:
    """Return decomposed campaign error metrics (A–D) plus standard campaign metrics."""
    n_gt = _n_ground_truth_campaigns(membership)
    accepted = _accepted_campaign_clusters(cluster_df)
    n_accepted = len(accepted)
    veh_col = resolve_vehicle_id_column(membership)

    # A. False campaign alert — predicted multi-vehicle campaign when no GT campaign exists.
    false_campaign_alert = int(n_gt == 0 and n_accepted > 0)

    # B. Benign membership contamination — benign vehicles in fleet campaign membership.
    if "fleet_campaign_member" in event_predictions.columns:
        fleet_mem = event_predictions.groupby(veh_col)["fleet_campaign_member"].max()
    else:
        fleet_mem = event_predictions.groupby(veh_col)["final_decision"].apply(
            lambda s: int((s == DECISION_COORDINATED).any())
        )
    gt_att = membership.groupby(veh_col)["ground_truth_campaign_member"].max()
    veh = pd.DataFrame({"gt": gt_att, "pred": fleet_mem}).fillna(0).astype(int)
    cm = compute_confusion(veh["gt"].to_numpy(), veh["pred"].to_numpy())
    benign_vehicles_included = int(cm["fp"])

    # C. Extra false campaign clusters beyond ground truth.
    if n_gt == 0:
        extra_cluster_count = n_accepted
        false_campaign_cluster_count = n_accepted
    else:
        extra_cluster_count = max(n_accepted - n_gt, 0)
        false_campaign_cluster_count = extra_cluster_count

    # D. Incorrect merging — unrelated incidents merged into one campaign.
    incorrect_merging = int(n_gt > 1 and n_accepted == 1)
    if scenario in ("V2", "S2") and n_gt > 1 and n_accepted == 1:
        incorrect_merging = 1

    # Fragmentation — one GT campaign split into multiple accepted clusters.
    fragmentation = int(n_gt > 0 and n_accepted > n_gt)

    # Legacy-compatible rates using corrected semantics.
    if n_gt == 0:
        false_campaign_alert_rate = float(false_campaign_alert)
        campaign_precision = 1.0 if n_accepted == 0 else 0.0
        campaign_recall = 1.0 if n_accepted == 0 else 0.0
    else:
        false_campaign_alert_rate = float(extra_cluster_count > 0 or false_campaign_alert)
        tp_clusters = min(n_accepted, n_gt)
        campaign_precision = _safe_div(tp_clusters, max(n_accepted, 1))
        campaign_recall = _safe_div(int(n_accepted >= 1), 1)

    campaign_f1 = _safe_div(2 * campaign_precision * campaign_recall, campaign_precision + campaign_recall)
    membership_precision = _safe_div(cm["tp"], cm["tp"] + cm["fp"])
    membership_recall = _safe_div(cm["tp"], cm["tp"] + cm["fn"])

    coordinated_events = int((event_predictions["final_decision"] == DECISION_COORDINATED).sum())

    return {
        "false_campaign_alert_indicator": false_campaign_alert,
        "false_campaign_cluster_count": false_campaign_cluster_count,
        "benign_vehicles_included": benign_vehicles_included,
        "extra_cluster_count": extra_cluster_count,
        "incorrect_merging": incorrect_merging,
        "fragmentation": fragmentation,
        "false_campaign_alert_rate": false_campaign_alert_rate,
        "campaign_precision": float(campaign_precision),
        "campaign_recall": float(campaign_recall),
        "campaign_f1": float(campaign_f1),
        "campaign_membership_precision": float(membership_precision),
        "campaign_membership_recall": float(membership_recall),
        "n_detected_campaign_clusters": n_accepted,
        "n_ground_truth_campaigns": n_gt,
        "coordinated_events": coordinated_events,
        "expect_campaign": expect_campaign,
    }


def legacy_false_campaign_rate_explanation(n_detected: int, expect_campaign: bool) -> dict[str, Any]:
    """Document why legacy metric equals n_detected/max(n_detected,1) when expect_campaign=True."""
    legacy = _safe_div(n_detected, max(n_detected + int(not expect_campaign), 1))
    return {
        "n_detected": n_detected,
        "expect_campaign": expect_campaign,
        "legacy_false_campaign_alert_rate": legacy,
        "legacy_always_one_when_detected": bool(expect_campaign and n_detected >= 1),
    }
