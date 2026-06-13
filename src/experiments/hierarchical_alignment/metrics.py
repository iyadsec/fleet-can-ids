"""Metrics for hierarchically aligned evaluation."""

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
from src.experiments.campaign_evaluation import compute_confusion, compute_campaign_metrics, _safe_div
from src.experiments.evaluation_correction.metrics import reconstruct_cluster_df
from src.experiments.hierarchical_alignment.transform import CONFIG_LABELS


def compute_local_event_metrics(
    events: pd.DataFrame,
    *,
    run_id: str,
    scenario_id: str,
    seed: int,
    use_weak_band: bool = True,
) -> dict[str, Any]:
    """Event metrics from Isolation Forest only (never fleet fields)."""
    y_true = events["ground_truth_malicious"].astype(int).to_numpy()
    y_pred = events["local_event_alert"].astype(int).to_numpy()
    cm = compute_confusion(y_true, y_pred)
    scores = pd.to_numeric(events["local_anomaly_score"], errors="coerce").fillna(0).to_numpy()
    roc = pr = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            roc = float(roc_auc_score(y_true, scores))
            pr = float(average_precision_score(y_true, scores))
        except ValueError:
            pass
    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "seed": seed,
        **cm,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": _safe_div(cm["fp"], cm["fp"] + cm["tn"]),
        "fnr": _safe_div(cm["fn"], cm["fn"] + cm["tp"]),
        "roc_auc": roc,
        "pr_auc": pr,
        "latency_sec": float("nan"),
        "prediction_source": "isolation_forest_only",
    }


def compute_fleet_campaign_run_metrics(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    run_id: str,
    method: str,
    scenario_id: str,
    seed: int,
    campaign_size: int,
    coordination_strength: float,
    expect_campaign: bool,
) -> dict[str, Any]:
    if method == "local_ids":
        return {
            "run_id": run_id,
            "method": method,
            "framework_config": "C1",
            "scenario_id": scenario_id,
            "seed": seed,
            "campaign_size": campaign_size,
            "coordination_strength": coordination_strength,
            "campaign_metrics_na": True,
        }
    cluster_df = reconstruct_cluster_df(events.rename(columns={"fleet_cluster_id": "cluster_id"}))
    if "final_decision" not in events.columns:
        events = events.copy()
        events["final_decision"] = np.where(
            events["fleet_campaign_member"] == 1, DECISION_COORDINATED, "isolated_attack"
        )
    camp = compute_campaign_metrics(events, membership, cluster_df, expect_campaign)
    n_qual = int(camp.get("n_detected_campaign_clusters", 0))
    n_gt = int(camp.get("n_ground_truth_campaigns", 0))
    frag = max(n_qual - max(n_gt, 1), 0) if n_gt else n_qual
    return {
        "run_id": run_id,
        "method": method,
        "framework_config": "C2" if method == "descriptor_clustering" else "C3",
        "scenario_id": scenario_id,
        "seed": seed,
        "campaign_size": campaign_size,
        "coordination_strength": coordination_strength,
        "campaign_metrics_na": False,
        "membership_purity": camp["campaign_precision"],
        "fragmentation": frag,
        "completeness": camp["campaign_recall"],
        "n_campaign_clusters": n_qual,
        **camp,
    }


def compute_weak_campaign_support(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    run_id: str,
    method: str,
    seed: int,
    campaign_size: int,
) -> dict[str, Any]:
    if method == "local_ids":
        return {"run_id": run_id, "method": method, "framework_config": "C1", "campaign_metrics_na": True}
    weak = events["local_evidence_level"] == "weak"
    gt_camp = events.get("ground_truth_campaign_id", pd.Series("", index=events.index)).astype(str).str.len() > 0
    gt_mal = events["ground_truth_malicious"] == 1
    fleet_mem = events["fleet_campaign_member"] == 1

    weak_camp_supported = int((weak & gt_camp & fleet_mem).sum())
    weak_attacked_veh = 0
    if "scenario_vehicle_id" in events.columns or "vehicle_token" in events.columns:
        veh_col = "scenario_vehicle_id" if "scenario_vehicle_id" in events.columns else "vehicle_token"
        attacked = membership[membership.get("ground_truth_campaign_member", 0) == 1][veh_col].unique()
        for v in attacked:
            g = events[events[veh_col] == v]
            if ((g["local_evidence_level"] == "weak") & (g["fleet_campaign_member"] == 1)).any():
                weak_attacked_veh += 1
    weak_benign_included = int((weak & (events["ground_truth_malicious"] == 0) & fleet_mem).sum())
    gt_weak_mal = int((weak & gt_mal).sum())
    return {
        "run_id": run_id,
        "method": method,
        "framework_config": "C2" if method == "descriptor_clustering" else "C3",
        "seed": seed,
        "campaign_size": campaign_size,
        "weak_campaign_members_supported": weak_camp_supported,
        "weak_attacked_vehicles_correlated": weak_attacked_veh,
        "weak_benign_signals_included": weak_benign_included,
        "weak_malicious_local_signals": gt_weak_mal,
        "weak_campaign_detection_rate": float(
            events[events["fleet_decision"] == "coordinated_campaign"]["event_id"].nunique() > 0
        ),
        "weak_campaign_membership_precision": _safe_div(
            int((fleet_mem & gt_camp).sum()), int(fleet_mem.sum())
        ),
    }


def compute_membership_errors(
    events: pd.DataFrame,
    *,
    run_id: str,
    method: str,
) -> dict[str, Any]:
    if method == "local_ids":
        return {"run_id": run_id, "method": method, "n_errors": 0}
    gt_camp = events.get("ground_truth_campaign_id", pd.Series("", index=events.index)).astype(str).str.len() > 0
    fleet_mem = events["fleet_campaign_member"] == 1
    fp = int((fleet_mem & ~gt_camp).sum())
    fn = int((~fleet_mem & gt_camp & (events["ground_truth_malicious"] == 1)).sum())
    return {
        "run_id": run_id,
        "method": method,
        "false_campaign_memberships": fp,
        "missed_campaign_memberships": fn,
        "n_errors": fp + fn,
    }


def capability_row(config: str) -> dict[str, Any]:
    if config == "C1":
        return {
            "configuration": CONFIG_LABELS["C1"],
            "local_anomaly_detection": "Yes",
            "weak_evidence_retained": "Yes",
            "isolated_incident_identification": "No",
            "campaign_identification": "N/A",
            "multi_vehicle_membership": "N/A",
            "raw_can_transmitted": "No (local features only)",
        }
    return {
        "configuration": CONFIG_LABELS.get(config, config),
        "local_anomaly_detection": "Yes (preserved)",
        "weak_evidence_retained": "Yes",
        "isolated_incident_identification": "Yes",
        "campaign_identification": "Yes",
        "multi_vehicle_membership": "Yes",
        "raw_can_transmitted": "No (descriptor graph only)",
    }


def build_capability_comparison() -> pd.DataFrame:
    c1 = capability_row("C1")
    c1["configuration"] = "Local-only IDS"
    c3 = capability_row("C3")
    c3["configuration"] = "Complete hierarchical framework"
    return pd.DataFrame([c1, c3]).rename(
        columns={
            "configuration": "Configuration",
            "local_anomaly_detection": "Local anomaly detection",
            "weak_evidence_retained": "Weak evidence retained",
            "isolated_incident_identification": "Isolated incident identification",
            "campaign_identification": "Campaign identification",
            "multi_vehicle_membership": "Multi-vehicle membership",
            "raw_can_transmitted": "Raw CAN transmitted",
        }
    )