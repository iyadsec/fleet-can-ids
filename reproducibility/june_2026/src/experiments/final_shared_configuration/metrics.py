"""Metrics extraction for final shared configuration runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.coordinated_campaign_refinement.candidates import candidate_statistics
from src.experiments.experiment_pipeline import resolve_vehicle_id_column
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown
from src.experiments.strong_campaign_extended.metrics import compute_diagnostic_row


def extract_run_metrics(
    *,
    run_id: str,
    scenario_id: str,
    attack_strength: str,
    seed: int,
    campaign_size: int,
    source_df: pd.DataFrame,
    membership: pd.DataFrame,
    outputs,
    cand_stats: dict[str, Any],
    expect_campaign: bool,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gs = outputs.graph_stats.iloc[0].to_dict() if not outputs.graph_stats.empty else {}
    rt = outputs.runtime
    diag = compute_diagnostic_row(
        run_id=run_id,
        seed=seed,
        campaign_size=campaign_size,
        fleet_size=20,
        total_nodes=int(cand_stats.get("source_windows", len(source_df))),
        experiment_group=scenario_id,
        method="fcgnn",
        event_predictions=outputs.event_predictions,
        vehicle_predictions=outputs.vehicle_predictions,
        membership=membership,
        cluster_df=outputs.cluster_df,
        graph_stats=gs,
        runtime=rt,
        base_metrics=base_metrics or {},
    )
    br = compute_false_campaign_breakdown(
        outputs.event_predictions, membership, outputs.cluster_df, expect_campaign=expect_campaign
    )
    row = {**diag, **cand_stats}
    row.update({
        "run_id": run_id,
        "scenario_id": scenario_id,
        "attack_strength": attack_strength,
        "seed": seed,
        "campaign_size": campaign_size,
        "expect_campaign": int(expect_campaign),
        "graph_nodes": int(gs.get("nodes", 0)),
        "candidate_transmission_rate": float(cand_stats.get("candidate_nodes", 0) / max(cand_stats.get("source_windows", 1), 1)),
        "transmission_percentage": 100.0 * float(cand_stats.get("candidate_nodes", 0) / max(cand_stats.get("source_windows", 1), 1)),
        "clearly_benign_filtered": int(cand_stats.get("correctly_filtered_benign", 0)),
        "benign_candidates": int(cand_stats.get("benign_candidates", 0)),
        "strong_candidates": int(cand_stats.get("strong_malicious_candidates", 0)),
        "weak_candidates": int(cand_stats.get("weak_malicious_candidates", 0)),
        "same_vehicle_edges": float(gs.get("same_vehicle_edges", np.nan)),
        "cross_vehicle_edges": float(gs.get("cross_vehicle_edges", np.nan)),
        "incorrect_merging": float(br.get("incorrect_merging", 0)),
        "false_campaign_rate": float(max(br.get("false_campaign_cluster_count", 0), 0) > 0) if not expect_campaign else np.nan,
        "local_detection_rate": float(
            (outputs.vehicle_predictions["predicted_attacked"] & outputs.vehicle_predictions["ground_truth_attacked"]).sum()
            / max(outputs.vehicle_predictions["ground_truth_attacked"].sum(), 1)
        ) if "predicted_attacked" in outputs.vehicle_predictions.columns else np.nan,
    })
    if not outputs.edge_list.empty and "source_vehicle" in outputs.edge_list.columns:
        cross = outputs.edge_list["source_vehicle"] != outputs.edge_list["target_vehicle"]
        row["malicious_cross_vehicle_edge_purity"] = np.nan
    return row


def safety_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "scenario_id": row["scenario_id"],
        "seed": row["seed"],
        "local_detection_rate": row.get("local_detection_rate"),
        "false_campaign_rate": row.get("false_campaign_rate"),
        "incorrect_merging": row.get("incorrect_merging"),
        "benign_candidate_rate": row.get("candidate_transmission_rate"),
        "correct_no_attack_rate": 1.0 - float(row.get("false_campaign_rate") or 0) if row["scenario_id"] == "S0" else np.nan,
    }
