"""Corrected metric aggregation for evaluation correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
from src.experiments.campaign_evaluation import (
    compute_campaign_metrics,
    compute_confusion,
    compute_event_metrics,
)
from src.experiments.vehicle_identity import resolve_vehicle_id_column


def reconstruct_cluster_df(event_predictions: pd.DataFrame) -> pd.DataFrame:
    """Rebuild cluster qualification table from saved event predictions."""
    if event_predictions.empty or "cluster_id" not in event_predictions.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for cid, grp in event_predictions.groupby("cluster_id"):
        cid_i = int(cid)
        if cid_i < 0:
            continue
        is_coord = bool((grp.get("final_decision", pd.Series()) == DECISION_COORDINATED).any())
        rows.append(
            {
                "cluster_id": cid_i,
                "is_qualifying_campaign_cluster": is_coord,
                "vehicles_in_cluster": int(
                    grp["vehicles_in_cluster"].max()
                    if "vehicles_in_cluster" in grp.columns
                    else grp.get("scenario_vehicle_id", grp.get("vehicle_token", pd.Series())).nunique()
                ),
                "behavioral_cohesion": float(grp["behavioral_cohesion"].max()) if "behavioral_cohesion" in grp.columns else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_vehicle_detailed_metrics(
    event_predictions: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, Any]:
    events = event_predictions.copy()
    veh_col = resolve_vehicle_id_column(events)
    if "ground_truth_campaign_member" not in events.columns:
        if "event_id" in membership.columns:
            mem_cols = [c for c in ("event_id", "ground_truth_campaign_member") if c in membership.columns]
            events = events.merge(membership[mem_cols].drop_duplicates("event_id"), on="event_id", how="left")
        else:
            mem_veh = resolve_vehicle_id_column(membership)
            veh_map = membership.groupby(mem_veh)["ground_truth_campaign_member"].max()
            events["ground_truth_campaign_member"] = events[veh_col].map(veh_map)
    mem_veh_col = resolve_vehicle_id_column(membership)
    gt_vehicles = (
        membership.groupby(mem_veh_col)["ground_truth_campaign_member"]
        .max()
        .reset_index()
        .rename(columns={mem_veh_col: "vehicle_id", "ground_truth_campaign_member": "ground_truth_attacked"})
    )
    pred_vehicles = (
        events.groupby(veh_col)
        .agg(
            predicted_attacked=("predicted_malicious", "max"),
            predicted_campaign=("predicted_campaign_membership", "max")
            if "predicted_campaign_membership" in event_predictions.columns
            else ("predicted_malicious", "max"),
            n_events=("event_id", "count"),
            n_pred_malicious=("predicted_malicious", "sum"),
            n_gt_malicious=("ground_truth_malicious", "sum"),
        )
        .reset_index()
        .rename(columns={veh_col: "vehicle_id"})
    )
    merged = gt_vehicles.merge(pred_vehicles, on="vehicle_id", how="left")
    y_true = merged["ground_truth_attacked"].astype(int).to_numpy()
    y_pred = merged["predicted_attacked"].astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    coverage_vals = []
    for vid, grp in events.groupby(veh_col):
        if int(grp["ground_truth_campaign_member"].max()) != 1:
            continue
        gt_m = int((grp["ground_truth_malicious"] == 1).sum())
        det_m = int(((grp["ground_truth_malicious"] == 1) & (grp["predicted_malicious"] == 1)).sum())
        coverage_vals.append(det_m / gt_m if gt_m else np.nan)

    return {
        "vehicle_precision": float(prec),
        "vehicle_recall": float(rec),
        "vehicle_f1": float(f1),
        "attacked_vehicles_recovered": tp,
        "benign_vehicles_incorrectly_included": fp,
        "vehicle_event_coverage_mean": float(np.nanmean(coverage_vals)) if coverage_vals else float("nan"),
        "vehicle_event_coverage_std": float(np.nanstd(coverage_vals)) if coverage_vals else float("nan"),
    }


def compute_event_confusion_row(
    event_predictions: pd.DataFrame,
    *,
    run_id: str,
    seed: int,
    method: str,
    attack_strength: str,
    campaign_size: int,
) -> dict[str, Any]:
    y_true = event_predictions["ground_truth_malicious"].astype(int)
    y_pred = event_predictions["predicted_malicious"].astype(int)
    cm = compute_confusion(y_true.to_numpy(), y_pred.to_numpy())
    metrics = compute_event_metrics(event_predictions)
    return {
        "run_id": run_id,
        "seed": seed,
        "method": method,
        "attack_strength": attack_strength,
        "campaign_size": campaign_size,
        "total_events": len(event_predictions),
        "total_malicious_events": int((y_true == 1).sum()),
        "total_benign_events": int((y_true == 0).sum()),
        "predicted_malicious_events": int((y_pred == 1).sum()),
        "predicted_benign_events": int((y_pred == 0).sum()),
        **cm,
        **{k: metrics[k] for k in ("precision", "recall", "f1", "fpr", "fnr", "weak_malicious_recovered", "weak_benign_promoted")},
        "original_fpr": metrics["fpr"],
    }


def _ensure_event_metric_columns(event_predictions: pd.DataFrame) -> pd.DataFrame:
    out = event_predictions.copy()
    if "anomaly_score" not in out.columns:
        out["anomaly_score"] = 0.0
    if "weak_signal" not in out.columns:
        out["weak_signal"] = 0
    if "final_decision" not in out.columns:
        out["final_decision"] = ""
    return out


def aggregate_corrected_run_metrics(
    event_predictions: pd.DataFrame,
    vehicle_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    method: str,
    seed: int,
    attack_strength: str,
    campaign_size: int,
    coordination_strength: float,
    runtime: dict[str, float],
    expect_campaign: bool,
) -> dict[str, Any]:
    event_m = compute_event_metrics(_ensure_event_metric_columns(event_predictions))
    if method == "local_ids":
        campaign_m = {
            "campaign_detection_rate": float("nan"),
            "campaign_precision": float("nan"),
            "campaign_recall": float("nan"),
            "campaign_f1": float("nan"),
            "false_campaign_alert_rate": float("nan"),
            "n_detected_campaign_clusters": 0,
            "n_ground_truth_campaigns": int(
                membership.loc[membership["ground_truth_campaign_id"].astype(str).str.len() > 0, "ground_truth_campaign_id"].nunique()
            ),
            "incorrect_campaign_merging": 0,
            "coordinated_events": 0,
            "campaign_metrics_na": True,
        }
    else:
        campaign_m = compute_campaign_metrics(
            event_predictions, membership, cluster_df, expect_campaign
        )
        campaign_m["campaign_metrics_na"] = False

    vehicle_m = compute_vehicle_detailed_metrics(event_predictions, membership)
    return {
        "method": method,
        "seed": seed,
        "attack_strength": attack_strength,
        "campaign_size": campaign_size,
        "coordination_strength": coordination_strength,
        **event_m,
        **vehicle_m,
        **campaign_m,
        **{f"runtime_{k}": v for k, v in runtime.items()},
    }


def campaign_error_breakdown_row(
    event_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    run_id: str,
    method: str,
    attack_strength: str,
    campaign_size: int,
    seed: int,
) -> dict[str, Any]:
    qualifying = (
        cluster_df[cluster_df["is_qualifying_campaign_cluster"]]
        if not cluster_df.empty and "is_qualifying_campaign_cluster" in cluster_df.columns
        else pd.DataFrame()
    )
    n_gt = membership.loc[
        membership["ground_truth_campaign_id"].astype(str).str.len() > 0, "ground_truth_campaign_id"
    ].nunique()
    n_detected = len(qualifying)
    camp = compute_campaign_metrics(event_predictions, membership, cluster_df, True)
    veh = compute_vehicle_detailed_metrics(event_predictions, membership)
    return {
        "run_id": run_id,
        "method": method,
        "attack_strength": attack_strength,
        "campaign_size": campaign_size,
        "seed": seed,
        "campaign_precision": camp["campaign_precision"],
        "campaign_recall": camp["campaign_recall"],
        "campaign_f1": camp["campaign_f1"],
        "campaign_detection_rate": camp["campaign_detection_rate"],
        "true_campaign_clusters": n_gt,
        "detected_campaign_clusters": n_detected,
        "false_campaign_clusters": max(n_detected - int(n_gt > 0), 0) if n_gt else n_detected,
        "missed_campaigns": int(n_gt > 0 and n_detected < 1),
        "benign_vehicles_incorrectly_included": veh["benign_vehicles_incorrectly_included"],
        "membership_purity_proxy": camp["campaign_precision"],
        "coordinated_events": camp.get("coordinated_events", 0),
        "incorrect_campaign_merging": camp.get("incorrect_campaign_merging", 0),
    }
