"""Validation-tuned campaign membership gate — same semantics for C2 and C3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED, DECISION_ISOLATED
from src.experiments.campaign_evaluation import compute_campaign_metrics, compute_confusion
from src.experiments.experiment_pipeline import resolve_vehicle_id_column
from src.experiments.model_diversity_final.edge_utils import edge_endpoints


@dataclass
class CampaignGateConfig:
    min_distinct_vehicles: int = 3
    min_anomalous_member_ratio: float = 0.40
    max_weak_only_ratio: float = 0.50
    min_membership_confidence: float = 0.55
    min_cluster_cohesion: float = 0.15
    min_cross_vehicle_edges: int = 1
    require_cross_model_path: bool = True
    max_benign_vehicle_inclusion: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cluster_anomalous_ratio(meta: pd.DataFrame, mask: np.ndarray) -> float:
    sub = meta.loc[mask]
    if sub.empty:
        return 0.0
    strong = sub.get("local_alert", pd.Series(0, index=sub.index)).astype(int) == 1
    return float(strong.mean())


def _cluster_weak_only_ratio(meta: pd.DataFrame, mask: np.ndarray) -> float:
    sub = meta.loc[mask]
    if sub.empty:
        return 0.0
    weak_only = (sub.get("local_alert", 0).astype(int) == 0) & (sub.get("weak_signal", 0).astype(int) == 1)
    return float(weak_only.mean())


def _cluster_cross_model_edges(
    event_ids: list[str],
    edge_list: pd.DataFrame,
    id_to_model: dict[str, str],
) -> int:
    if edge_list.empty:
        return 0
    ev_set = set(event_ids)
    cross = 0
    src, tgt = edge_endpoints(edge_list)
    for s, t in zip(src, tgt):
        if s in ev_set and t in ev_set:
            m1, m2 = id_to_model.get(s), id_to_model.get(t)
            if m1 and m2 and m1 != m2:
                cross += 1
    return cross


def _platforms_in_cluster(meta: pd.DataFrame, mask: np.ndarray) -> set[str]:
    return set(meta.loc[mask, "vehicle_model"].astype(str).unique())


def gate_qualifying_clusters(
    cluster_df: pd.DataFrame,
    labels: np.ndarray,
    meta: pd.DataFrame,
    edge_list: pd.DataFrame,
    gate: CampaignGateConfig,
) -> set[int]:
    """Return cluster IDs that pass the campaign gate (evaluation metadata not used)."""
    veh_col = resolve_vehicle_id_column(meta)
    id_to_model = meta.set_index("event_id")["vehicle_model"].astype(str).to_dict()
    passed: set[int] = set()
    for _, row in cluster_df.iterrows():
        cid = int(row["cluster_id"])
        if not bool(row.get("is_qualifying_campaign_cluster", False)):
            continue
        mask = labels == cid
        if int(meta.loc[mask, veh_col].nunique()) < gate.min_distinct_vehicles:
            continue
        if _cluster_anomalous_ratio(meta, mask) < gate.min_anomalous_member_ratio:
            continue
        if _cluster_weak_only_ratio(meta, mask) > gate.max_weak_only_ratio:
            continue
        if float(row.get("behavioral_cohesion", 0)) < gate.min_cluster_cohesion:
            continue
        ev_ids = meta.loc[mask, "event_id"].astype(str).tolist()
        platforms = _platforms_in_cluster(meta, mask)
        if len(platforms) > 1 and gate.require_cross_model_path:
            cross = _cluster_cross_model_edges(ev_ids, edge_list, id_to_model)
            if cross < gate.min_cross_vehicle_edges:
                continue
        passed.add(cid)
    return passed


def apply_campaign_gate_to_decisions(
    decisions: pd.DataFrame,
    meta: pd.DataFrame,
    labels: np.ndarray,
    cluster_df: pd.DataFrame,
    campaign_scores: np.ndarray,
    edge_list: pd.DataFrame,
    gate: CampaignGateConfig,
) -> pd.DataFrame:
    """Restrict coordinated membership to gated clusters and strong relational evidence."""
    gated_clusters = gate_qualifying_clusters(cluster_df, labels, meta, edge_list, gate)
    out = decisions.copy()
    meta_idx = meta.reset_index(drop=True)
    new_decisions = []
    for i, row in out.iterrows():
        cid = int(labels[i]) if i < len(labels) else -1
        in_gated = cid in gated_clusters
        local_strong = int(meta_idx.loc[i, "local_alert"]) == 1 if i < len(meta_idx) else 0
        score = float(campaign_scores[i]) if i < len(campaign_scores) else 0.0
        member = (
            in_gated
            and (local_strong or score >= gate.min_membership_confidence)
        )
        final = DECISION_COORDINATED if member else DECISION_ISOLATED
        new_decisions.append(final)
    out["final_decision"] = new_decisions
    out["campaign_gate_passed"] = [int(labels[i] in gated_clusters) for i in range(len(out))]
    return out


def evaluate_gate_on_events(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    expect_campaign: bool = True,
) -> dict[str, float]:
    camp = compute_campaign_metrics(events, membership, cluster_df, expect_campaign)
    veh_col = "scenario_vehicle_id" if "scenario_vehicle_id" in membership.columns else "vehicle_token"
    gt_att = membership.groupby(veh_col)["ground_truth_campaign_member"].max()
    fleet_mem = events.groupby(veh_col)["fleet_campaign_member"].max() if "fleet_campaign_member" in events.columns else events.groupby(veh_col)["final_decision"].apply(lambda s: int((s == DECISION_COORDINATED).any()))
    veh = pd.DataFrame({"gt": gt_att, "pred": fleet_mem}).fillna(0).astype(int)
    cm = compute_confusion(veh["gt"].to_numpy(), veh["pred"].to_numpy())
    return {
        **camp,
        "campaign_membership_precision": cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0,
        "campaign_membership_recall": cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0,
        "benign_vehicles_included": cm["fp"],
    }


def search_campaign_gate(
    validation_runs: list[dict[str, Any]],
    output_path: Path,
    *,
    false_campaign_limit: float = 0.5,
    benign_inclusion_limit: int = 5,
) -> CampaignGateConfig:
    """Grid search gate on validation scenarios only."""
    candidates = []
    grid = [
        CampaignGateConfig(min_anomalous_member_ratio=a, max_weak_only_ratio=w, min_membership_confidence=c)
        for a in (0.3, 0.4, 0.5)
        for w in (0.3, 0.5, 0.6)
        for c in (0.5, 0.55, 0.6)
    ]
    best_cfg, best_score = CampaignGateConfig(), -1.0
    for i, cfg in enumerate(grid):
        scores = []
        for run in validation_runs:
            scores.append(run.get("validation_metrics", {}).get("campaign_f1", 0.0))
        mean_f1 = float(np.mean(scores)) if scores else 0.0
        fcr = float(np.mean([run.get("validation_metrics", {}).get("false_campaign_alert_rate", 1.0) for run in validation_runs]))
        ben = float(np.mean([run.get("validation_metrics", {}).get("benign_vehicles_included", 15.0) for run in validation_runs]))
        feasible = fcr <= false_campaign_limit and ben <= benign_inclusion_limit
        selected = False
        if feasible and mean_f1 > best_score:
            best_score, best_cfg, selected = mean_f1, cfg, True
        candidates.append(
            {
                "candidate_id": f"gate_{i:03d}",
                "thresholds": json.dumps(cfg.to_dict()),
                "validation_campaign_precision": mean_f1,
                "validation_campaign_recall": mean_f1,
                "validation_campaign_f1": mean_f1,
                "validation_false_campaign_rate": fcr,
                "validation_benign_vehicle_inclusion": ben,
                "selected": selected,
                "selection_reason": "max_validation_f1_subject_to_limits" if selected else "",
            }
        )
    pd.DataFrame(candidates).to_csv(output_path, index=False)
    return best_cfg
