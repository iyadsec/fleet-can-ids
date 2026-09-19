"""Fleet campaign decision via shared publication FLEET-GUARD path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.fleet_graph import fit_or_load_ctt_scaler, resolve_ctt_fleet_config
from src.ctt.utils import ensure_dir, safe_div, write_markdown
from src.evaluation.publication_fleet_core import (
    FleetRunArtifacts,
    PublicationFleetConfig,
    match_predicted_to_gt_campaigns_jaccard,
    run_publication_fleet_pipeline,
)
from src.experiments.local_descriptor_normalisation import FleetScalerProvenance


def run_fleet_campaign_inference(
    desc_df: pd.DataFrame,
    *,
    scaler: FleetScalerProvenance | None = None,
    cfg: PublicationFleetConfig | None = None,
    seed: int = 42,
) -> FleetRunArtifacts:
    """
    Model inference only — no GT attack labels/types/campaign membership as inputs.

    Attack-type columns may be present on ``desc_df`` for post-hoc evaluation
    fields inside the shared cluster summary, but they are not used by the gate.
    """
    cfg = cfg or resolve_ctt_fleet_config(seed=seed)
    if seed != cfg.seed:
        cfg = PublicationFleetConfig(**{**cfg.to_dict(), "seed": seed})
    scaler = scaler or fit_or_load_ctt_scaler(desc_df)
    return run_publication_fleet_pipeline(
        desc_df, fleet_scaler_provenance=scaler, cfg=cfg
    )


def evaluate_campaign(
    artifacts: FleetRunArtifacts,
    scenario_type: str,
    *,
    ground_truth_campaign_vehicles: set[str] | None = None,
    ground_truth_campaigns: list[set[str]] | None = None,
) -> dict[str, Any]:
    """
    Post-hoc evaluation only.

    Matching rule (documented; P7/P8 publication matcher unrecovered):
    greedy Jaccard ≥ 0.5 between predicted qualifying campaign vehicle sets and
    GT campaign vehicle sets (ablation peer). When a single GT vehicle set is
    provided, membership metrics use that set against the best-matching cluster.
    """
    summary = artifacts.cluster_summary
    decisions = artifacts.node_decisions
    qualifying = (
        summary[summary["is_qualifying_campaign_cluster"]]
        if not summary.empty
        else pd.DataFrame()
    )
    campaign_detected = not qualifying.empty

    pred_sets: list[set[str]] = []
    if campaign_detected and not decisions.empty:
        for cid in qualifying["cluster_id"].astype(int):
            members = decisions[decisions["cluster_id"] == cid]
            veh_col = "vehicle_token" if "vehicle_token" in members.columns else "vehicle_id"
            pred_sets.append(set(members[veh_col].astype(str)))

    gt_sets = list(ground_truth_campaigns or [])
    if ground_truth_campaign_vehicles and not gt_sets:
        gt_sets = [set(ground_truth_campaign_vehicles)]

    matches = match_predicted_to_gt_campaigns_jaccard(pred_sets, gt_sets) if gt_sets else []

    # Campaign-level precision/recall over matched campaigns
    if gt_sets:
        tp = len(matches)
        campaign_precision = safe_div(tp, len(pred_sets))
        campaign_recall = safe_div(tp, len(gt_sets))
    else:
        campaign_precision = 0.0
        campaign_recall = 0.0
    campaign_f1 = safe_div(
        2 * campaign_precision * campaign_recall,
        campaign_precision + campaign_recall,
    )

    # Membership metrics on best match (or sole GT set)
    membership_precision = membership_recall = membership_f1 = 0.0
    if matches and gt_sets:
        pi, gi, _ = matches[0]
        detected = pred_sets[pi]
        gt = gt_sets[gi]
        tp_m = len(detected & gt)
        membership_precision = safe_div(tp_m, len(detected))
        membership_recall = safe_div(tp_m, len(gt))
        membership_f1 = safe_div(
            2 * membership_precision * membership_recall,
            membership_precision + membership_recall,
        )
    elif ground_truth_campaign_vehicles and campaign_detected and pred_sets:
        # Fallback: largest predicted set vs GT vehicles
        detected = max(pred_sets, key=len)
        gt = set(ground_truth_campaign_vehicles)
        tp_m = len(detected & gt)
        membership_precision = safe_div(tp_m, len(detected))
        membership_recall = safe_div(tp_m, len(gt))
        membership_f1 = safe_div(
            2 * membership_precision * membership_recall,
            membership_precision + membership_recall,
        )

    false_campaign = False
    if scenario_type == "benign_fleet_control" and campaign_detected:
        false_campaign = True
    if scenario_type == "isolated_attack" and campaign_detected:
        false_campaign = any(len(s) > 1 for s in pred_sets)

    incorrect_merging = 0.0
    if scenario_type == "unrelated_incidents" and campaign_detected:
        # Incorrect merge if a qualifying cluster spans >1 distinct GT attack groups
        incorrect_merging = 1.0 if any(len(s) > 1 for s in pred_sets) else 0.0

    n_clusters = int(summary["cluster_id"].nunique()) if not summary.empty else 0
    n_assigned = int((decisions["cluster_id"] >= 0).sum()) if not decisions.empty else 0
    fragmentation = max(n_assigned - n_clusters, 0) if n_clusters else 0

    return {
        "scenario_type": scenario_type,
        "campaign_detected": int(campaign_detected),
        "false_campaign": int(false_campaign),
        "incorrect_merging": float(incorrect_merging),
        "false_campaign_rate": float(false_campaign),
        "campaign_precision": float(campaign_precision),
        "campaign_recall": float(campaign_recall),
        "campaign_f1": float(campaign_f1),
        "membership_precision": float(membership_precision),
        "membership_recall": float(membership_recall),
        "membership_f1": float(membership_f1),
        "fragmentation": int(fragmentation),
        "n_qualifying_campaigns": int(len(qualifying)),
        "n_clusters": n_clusters,
        "matching_rule": "greedy_jaccard_ge_0.5_ablation_peer",
        **{f"graph_{k}": v for k, v in artifacts.graph_stats.items() if not isinstance(v, list)},
    }


def write_fleet_transfer_policy(output_root: Path = OUTPUT_ROOT) -> None:
    ensure_dir(output_root / "audit")
    write_markdown(
        output_root / "audit" / "fleet_model_transfer_policy.md",
        "Fleet Model Transfer Policy",
        {
            "Decision": (
                "Shared publication FLEET-GUARD fleet path "
                "(src.evaluation.publication_fleet_core) with frozen OCSLab config"
            ),
            "Rationale": (
                "CTT reuses the same GraphSAGE structure training, constrained-kNN graph, "
                "StandardScaler→PCA→euclidean DBSCAN, and centroid cohesion campaign gate "
                "as the authoritative balanced OCSLab publication experiment. "
                "Only dataset loading and controlled scenario construction remain CTT-specific."
            ),
            "Temporal edges": "None — all edges are behavioural similarity only.",
            "Label leakage": (
                "Attack labels/types/campaign membership are evaluation-only; "
                "not used as GraphSAGE inputs, training targets, clustering inputs, "
                "or campaign-gate features."
            ),
        },
    )
