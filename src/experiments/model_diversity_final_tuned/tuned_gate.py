"""Two-level campaign gate: campaign acceptance + member acceptance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED, DECISION_ISOLATED
from src.experiments.experiment_pipeline import resolve_vehicle_id_column
from src.experiments.model_diversity_final.edge_utils import edge_endpoints


@dataclass
class TunedGateConfig:
    """Shared semantic gate for C2 and C3; numeric thresholds from validation only."""

    # Campaign-level acceptance
    min_distinct_vehicles: int = 3
    min_anomalous_ratio: float = 0.50
    max_weak_only_ratio: float = 0.50
    min_cluster_cohesion: float = 0.15
    min_cross_vehicle_support: int = 1
    min_cross_model_edges: int = 0
    max_benign_support_ratio: float = 0.30
    min_connected_platforms: int = 1
    require_cross_model_path: bool = True

    # Member-level acceptance
    min_membership_confidence: float = 0.55
    min_cross_vehicle_neighbors: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TunedGateConfig:
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})


def _cluster_anomalous_ratio(meta: pd.DataFrame, mask: np.ndarray) -> float:
    sub = meta.loc[mask]
    if sub.empty:
        return 0.0
    return float((sub.get("local_alert", 0).astype(int) == 1).mean())


def _cluster_weak_only_ratio(meta: pd.DataFrame, mask: np.ndarray) -> float:
    sub = meta.loc[mask]
    if sub.empty:
        return 0.0
    weak_only = (sub.get("local_alert", 0).astype(int) == 0) & (sub.get("weak_signal", 0).astype(int) == 1)
    return float(weak_only.mean())


def _cluster_benign_support_ratio(meta: pd.DataFrame, mask: np.ndarray) -> float:
    sub = meta.loc[mask]
    if sub.empty:
        return 0.0
    benign = sub.get("ground_truth_campaign_member", sub.get("scenario_gt_malicious", 0)).astype(int) == 0
    return float(benign.mean())


def _cross_model_edge_count(event_ids: list[str], edge_list: pd.DataFrame, id_to_model: dict[str, str]) -> int:
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


def _cross_vehicle_neighbor_count(
    event_id: str,
    cluster_event_ids: set[str],
    edge_list: pd.DataFrame,
    id_to_vehicle: dict[str, str],
) -> int:
    if edge_list.empty:
        return 0
    src, tgt = edge_endpoints(edge_list)
    neighbors: set[str] = set()
    for s, t in zip(src, tgt):
        if s == event_id and t in cluster_event_ids:
            neighbors.add(id_to_vehicle.get(t, ""))
        elif t == event_id and s in cluster_event_ids:
            neighbors.add(id_to_vehicle.get(s, ""))
    neighbors.discard(id_to_vehicle.get(event_id, ""))
    return len({v for v in neighbors if v})


def accept_campaign_clusters(
    cluster_df: pd.DataFrame,
    labels: np.ndarray,
    meta: pd.DataFrame,
    edge_list: pd.DataFrame,
    gate: TunedGateConfig,
) -> set[int]:
    """Campaign-level acceptance gate — independent of member filtering."""
    veh_col = resolve_vehicle_id_column(meta)
    id_to_model = meta.set_index("event_id")["vehicle_model"].astype(str).to_dict()
    passed: set[int] = set()
    for _, row in cluster_df.iterrows():
        if not bool(row.get("is_qualifying_campaign_cluster", False)):
            continue
        cid = int(row["cluster_id"])
        mask = labels == cid
        n_veh = int(meta.loc[mask, veh_col].nunique())
        if n_veh < gate.min_distinct_vehicles:
            continue
        if _cluster_anomalous_ratio(meta, mask) < gate.min_anomalous_ratio:
            continue
        if _cluster_weak_only_ratio(meta, mask) > gate.max_weak_only_ratio:
            continue
        if float(row.get("behavioral_cohesion", 0)) < gate.min_cluster_cohesion:
            continue
        if _cluster_benign_support_ratio(meta, mask) > gate.max_benign_support_ratio:
            continue
        platforms = set(meta.loc[mask, "vehicle_model"].astype(str).unique())
        if len(platforms) < gate.min_connected_platforms:
            continue
        ev_ids = meta.loc[mask, "event_id"].astype(str).tolist()
        cross_model = _cross_model_edge_count(ev_ids, edge_list, id_to_model)
        if cross_model < gate.min_cross_model_edges:
            continue
        if len(platforms) > 1 and gate.require_cross_model_path and cross_model < gate.min_cross_vehicle_support:
            continue
        passed.add(cid)
    return passed


def accept_cluster_members(
    meta: pd.DataFrame,
    labels: np.ndarray,
    accepted_clusters: set[int],
    campaign_scores: np.ndarray,
    edge_list: pd.DataFrame,
    gate: TunedGateConfig,
) -> list[bool]:
    """Member-level acceptance — may reject low-confidence benign members of accepted clusters."""
    veh_col = resolve_vehicle_id_column(meta)
    meta_r = meta.reset_index(drop=True)
    id_to_vehicle = meta_r.set_index("event_id")[veh_col].astype(str).to_dict()
    members: list[bool] = []
    for i, row in meta_r.iterrows():
        cid = int(labels[i])
        if cid not in accepted_clusters:
            members.append(False)
            continue
        cluster_mask = labels == cid
        cluster_ev = set(meta_r.loc[cluster_mask, "event_id"].astype(str))
        local_strong = int(row.get("local_alert", 0)) == 1
        score = float(campaign_scores[i]) if i < len(campaign_scores) else 0.0
        cross_neigh = _cross_vehicle_neighbor_count(
            str(row["event_id"]), cluster_ev, edge_list, id_to_vehicle
        )
        is_benign = int(row.get("ground_truth_campaign_member", row.get("scenario_gt_malicious", 0))) == 0
        weak_only = int(row.get("local_alert", 0)) == 0 and int(row.get("weak_signal", 0)) == 1
        if cross_neigh < gate.min_cross_vehicle_neighbors and not local_strong:
            members.append(False)
            continue
        if is_benign and weak_only and score < gate.min_membership_confidence:
            members.append(False)
            continue
        member = local_strong or score >= gate.min_membership_confidence
        members.append(bool(member))
    return members


def apply_tuned_gate(
    meta: pd.DataFrame,
    labels: np.ndarray,
    cluster_df: pd.DataFrame,
    campaign_scores: np.ndarray,
    edge_list: pd.DataFrame,
    gate: TunedGateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply two-level gate; annotate cluster_df with campaign_accepted."""
    accepted = accept_campaign_clusters(cluster_df, labels, meta, edge_list, gate)
    member_flags = accept_cluster_members(meta, labels, accepted, campaign_scores, edge_list, gate)

    out_rows = []
    meta_r = meta.reset_index(drop=True)
    cluster_info = cluster_df.set_index("cluster_id") if not cluster_df.empty else pd.DataFrame()
    for i, row in meta_r.iterrows():
        cid = int(labels[i])
        in_campaign = cid in accepted and member_flags[i]
        info = cluster_info.loc[cid] if cid in cluster_info.index else None
        out_rows.append(
            {
                "window_id": int(row["window_id"]),
                "vehicle_id": row["vehicle_model"],
                "vehicle_model": row["vehicle_model"],
                "event_id": row["event_id"],
                "attack_type": row["attack_type"],
                "anomaly_score": float(row["anomaly_score"]),
                "local_alert": int(row.get("local_alert", 0)),
                "gnn_campaign_score": round(float(campaign_scores[i]), 4),
                "cluster_id": cid,
                "vehicles_in_cluster": int(info["vehicles_in_cluster"]) if info is not None else 0,
                "behavioral_cohesion": round(float(info["behavioral_cohesion"]), 4) if info is not None else 0.0,
                "eval_dominant_attack_type": str(info["eval_dominant_attack_type"]) if info is not None else "",
                "final_decision": DECISION_COORDINATED if in_campaign else DECISION_ISOLATED,
                "campaign_gate_passed": int(cid in accepted),
                "member_gate_passed": int(member_flags[i]),
            }
        )
    decisions = pd.DataFrame(out_rows)
    cluster_out = cluster_df.copy()
    if not cluster_out.empty:
        cluster_out["campaign_accepted"] = cluster_out["cluster_id"].astype(int).isin(accepted)
    return decisions, cluster_out


def gate_parameter_grid(
    cohesion_values: list[float],
    confidence_values: list[float],
) -> list[TunedGateConfig]:
    """Full validation grid search parameter combinations."""
    configs: list[TunedGateConfig] = []
    for mdv in (2, 3, 4):
        for mar in (0.40, 0.50, 0.60, 0.70, 0.80):
            for mwor in (0.25, 0.50, 0.75, 1.00):
                for cohesion in cohesion_values:
                    for conf in confidence_values:
                        for mcvs in (1, 2, 3, 4):
                            for mcme in (0, 1, 2):
                                for mbsr in (0.10, 0.20, 0.30, 0.40):
                                    for mcp in (1, 2):
                                        configs.append(
                                            TunedGateConfig(
                                                min_distinct_vehicles=mdv,
                                                min_anomalous_ratio=mar,
                                                max_weak_only_ratio=mwor,
                                                min_cluster_cohesion=cohesion,
                                                min_membership_confidence=conf,
                                                min_cross_vehicle_support=mcvs,
                                                min_cross_model_edges=mcme,
                                                max_benign_support_ratio=mbsr,
                                                min_connected_platforms=mcp,
                                            )
                                        )
    return configs
