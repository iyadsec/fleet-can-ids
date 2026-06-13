"""M1 — Local per-vehicle Isolation Forest IDS baseline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_ISOLATED
from src.experiments.experiment_pipeline import MethodOutputs, resolve_vehicle_id_column
from src.experiments.result_writer import ExperimentRunContext


def run_local_ids_method(
    ctx: ExperimentRunContext,
    scenario_records: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
) -> MethodOutputs:
    """
    Local IDS only — no graph, clustering, or GNN.

    Campaign baseline: multi-vehicle *alert presence* when >= N vehicles fire
    local_alert; this is NOT behavioural campaign identification.
    """
    local_cfg = config.get("local_ids", {})
    min_vehicles = int(local_cfg.get("local_alert_min_vehicles", 2))

    id_cols = [c for c in ("vehicle_token", "scenario_vehicle_id") if c in membership.columns]
    if id_cols:
        df = scenario_records.merge(
            membership[["event_id", *id_cols]].drop_duplicates(subset=["event_id"]),
            on="event_id",
            how="left",
            suffixes=("", "_mem"),
        )
    else:
        df = scenario_records.copy()
    df["predicted_malicious"] = (
        (df["local_alert"] == 1) | (df.get("weak_signal", 0) == 1)
    ).astype(int)
    df["final_decision"] = DECISION_ISOLATED
    df["cluster_id"] = -1
    df["method"] = "local_ids"

    veh_col = resolve_vehicle_id_column(df)
    alerted_vehicles = df.loc[df["local_alert"] == 1, veh_col].nunique()
    multi_vehicle_alert = int(alerted_vehicles >= min_vehicles)

    event_pred = df.merge(
        membership[
            [
                "event_id",
                "ground_truth_malicious",
                "ground_truth_campaign_id",
                "scenario_role",
            ]
        ],
        on="event_id",
        how="left",
        suffixes=("", "_mem"),
    )
    if event_pred["ground_truth_malicious"].isna().any() and "scenario_gt_malicious" in event_pred.columns:
        event_pred["ground_truth_malicious"] = event_pred["ground_truth_malicious"].fillna(
            event_pred["scenario_gt_malicious"]
        )
    event_pred["method"] = "local_ids"
    if "anomaly_score" not in event_pred.columns:
        event_pred["anomaly_score"] = 0.0
    if "weak_signal" not in event_pred.columns:
        event_pred["weak_signal"] = 0
    event_pred["multi_vehicle_local_alert"] = multi_vehicle_alert

    veh_col = resolve_vehicle_id_column(event_pred)
    vehicle_pred = (
        event_pred.groupby(veh_col, as_index=False)
        .agg(
            predicted_attacked=("predicted_malicious", "max"),
            ground_truth_attacked=("ground_truth_malicious", "max"),
            n_events=("event_id", "count"),
            n_local_alerts=("local_alert", "sum"),
        )
        .rename(columns={veh_col: "vehicle_id"})
    )
    if "vehicle_model" in event_pred.columns:
        model_map = event_pred.groupby(veh_col)["vehicle_model"].first()
        vehicle_pred["vehicle_model"] = vehicle_pred["vehicle_id"].map(model_map)

    campaign_pred = pd.DataFrame(
        [
            {
                "method": "local_ids",
                "multi_vehicle_local_alert_presence": multi_vehicle_alert,
                "coordinated_campaign_identified": 0,
                "n_qualifying_campaign_clusters": 0,
                "n_alerted_vehicles": alerted_vehicles,
            }
        ]
    )

    return MethodOutputs(
        event_predictions=event_pred,
        vehicle_predictions=vehicle_pred,
        campaign_predictions=campaign_pred,
        cluster_df=pd.DataFrame(),
        embeddings=None,
        graph_stats=pd.DataFrame(),
        edge_list=pd.DataFrame(),
        runtime={"local_ids_sec": 0.0},
    )
