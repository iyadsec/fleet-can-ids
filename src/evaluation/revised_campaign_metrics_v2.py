"""Revised campaign metric protocol v2 (vehicle-id sets only).

Protocol id: metric_protocol=revised_jaccard_0.5_v2

Implements matching / membership / incorrect-merge / fragmentation logic for
prospective η-revalidation. Does NOT claim historical P7/P8 identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

VehicleId = Hashable
CampaignId = Hashable

TAU_J = 0.5
METRIC_PROTOCOL_ID = "revised_jaccard_0.5_v3"
METRIC_PROTOCOL_ID_V2 = "revised_jaccard_0.5_v2"


def jaccard(a: Iterable[VehicleId], b: Iterable[VehicleId]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


@dataclass(frozen=True)
class PredictedCampaign:
    """Gate-passing predicted campaign (noise already excluded)."""

    campaign_id: CampaignId
    vehicles: frozenset[VehicleId]


@dataclass(frozen=True)
class GroundTruthCampaign:
    campaign_id: CampaignId
    vehicles: frozenset[VehicleId]


@dataclass
class MatchPair:
    predicted_id: CampaignId
    gt_id: CampaignId
    jaccard: float
    tp_vehicle: int
    fp_vehicle: int
    fn_vehicle: int
    membership_precision: float
    membership_recall: float
    membership_f1: float


@dataclass
class ProtocolV2Result:
    metric_protocol: str = METRIC_PROTOCOL_ID_V2
    tau_j: float = TAU_J
    n_predicted: int = 0
    n_gt: int = 0
    tp_campaign: int = 0
    fp_campaign: int = 0
    fn_campaign: int = 0
    campaign_precision: float = 0.0
    campaign_recall: float = 0.0
    campaign_f1: float = 0.0
    false_campaign: bool = False
    matches: list[MatchPair] = field(default_factory=list)
    unmatched_predicted_ids: list[CampaignId] = field(default_factory=list)
    unmatched_gt_ids: list[CampaignId] = field(default_factory=list)
    unmatched_predicted_vehicle_count: int = 0
    # Micro membership
    tp_vehicle_sum: int = 0
    fp_vehicle_sum: int = 0
    fn_vehicle_sum: int = 0
    membership_precision_micro: float = 0.0
    membership_recall_micro: float = 0.0
    membership_f1_micro: float = 0.0
    # V2-only optional audit MemP (removed as primary in V3)
    membership_precision_including_unmatched_pred: float = 0.0
    incorrect_merge_run: bool = False
    incorrect_merge_predicted_ids: list[CampaignId] = field(default_factory=list)
    fragments_per_gt: dict[CampaignId, int] = field(default_factory=dict)
    fragmentation_per_gt: dict[CampaignId, int] = field(default_factory=dict)
    fragmentation_rate_mean: float = 0.0


# Alias for V3 callers
ProtocolV3Result = ProtocolV2Result


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, p + r)


def predicted_campaigns_from_clusters(
    cluster_to_vehicles: Mapping[CampaignId, Iterable[VehicleId]],
    *,
    noise_label: CampaignId = -1,
) -> list[PredictedCampaign]:
    """Build predicted campaigns, excluding DBSCAN noise label."""
    out: list[PredictedCampaign] = []
    for cid, vehicles in cluster_to_vehicles.items():
        if cid == noise_label:
            continue
        vs = frozenset(vehicles)
        if not vs:
            continue
        out.append(PredictedCampaign(campaign_id=cid, vehicles=vs))
    return out


def maximum_weight_jaccard_assignment(
    predicted: Sequence[PredictedCampaign],
    gt: Sequence[GroundTruthCampaign],
    *,
    tau_j: float = TAU_J,
) -> list[tuple[int, int, float]]:
    """Return list of (pred_index, gt_index, jaccard) matched pairs.

    Maximum-weight one-to-one assignment over pairs with J >= tau_j.
    """
    n_p, n_g = len(predicted), len(gt)
    if n_p == 0 or n_g == 0:
        return []

    weights = np.zeros((n_p, n_g), dtype=np.float64)
    for i, p in enumerate(predicted):
        for j, g in enumerate(gt):
            jacc = jaccard(p.vehicles, g.vehicles)
            if jacc >= tau_j:
                weights[i, j] = jacc

    # Maximize sum of weights via minimizing negative weights.
    # Ineligible / zero-weight edges stay 0 and are dropped after assignment.
    row_ind, col_ind = linear_sum_assignment(-weights)
    matches: list[tuple[int, int, float]] = []
    for i, j in zip(row_ind, col_ind):
        w = float(weights[i, j])
        if w >= tau_j:
            matches.append((int(i), int(j), w))
    return matches


def incorrect_merge_flags(
    predicted: Sequence[PredictedCampaign],
    incident_of: Mapping[VehicleId, CampaignId],
) -> list[CampaignId]:
    """Predicted campaigns whose attacked vehicles span >= 2 GT incidents."""
    bad: list[CampaignId] = []
    for p in predicted:
        incidents = {
            incident_of[v]
            for v in p.vehicles
            if v in incident_of
        }
        if len(incidents) >= 2:
            bad.append(p.campaign_id)
    return bad


def fragments_of_gt(
    predicted: Sequence[PredictedCampaign],
    gt: GroundTruthCampaign,
) -> int:
    """Number of predicted campaigns that intersect GT vehicle set."""
    return sum(1 for p in predicted if len(p.vehicles & gt.vehicles) > 0)


def evaluate_protocol_v2(
    predicted: Sequence[PredictedCampaign],
    gt_campaigns: Sequence[GroundTruthCampaign],
    *,
    incident_of: Mapping[VehicleId, CampaignId] | None = None,
    tau_j: float = TAU_J,
) -> ProtocolV2Result:
    """Evaluate one scenario run under revised_jaccard_0.5_v2 (archived semantics)."""
    return _evaluate_protocol(
        predicted,
        gt_campaigns,
        incident_of=incident_of,
        tau_j=tau_j,
        protocol_id=METRIC_PROTOCOL_ID_V2,
        include_unmatched_pred_in_primary_mem_fp=False,
    )


def evaluate_protocol_v3(
    predicted: Sequence[PredictedCampaign],
    gt_campaigns: Sequence[GroundTruthCampaign],
    *,
    incident_of: Mapping[VehicleId, CampaignId] | None = None,
    tau_j: float = TAU_J,
) -> ProtocolV3Result:
    """Evaluate one scenario run under revised_jaccard_0.5_v3 (current proposal)."""
    return _evaluate_protocol(
        predicted,
        gt_campaigns,
        incident_of=incident_of,
        tau_j=tau_j,
        protocol_id=METRIC_PROTOCOL_ID,
        include_unmatched_pred_in_primary_mem_fp=True,
    )


def _evaluate_protocol(
    predicted: Sequence[PredictedCampaign],
    gt_campaigns: Sequence[GroundTruthCampaign],
    *,
    incident_of: Mapping[VehicleId, CampaignId] | None,
    tau_j: float,
    protocol_id: str,
    include_unmatched_pred_in_primary_mem_fp: bool,
) -> ProtocolV2Result:
    pred = list(predicted)
    gt = list(gt_campaigns)
    result = ProtocolV2Result(
        metric_protocol=protocol_id, n_predicted=len(pred), n_gt=len(gt), tau_j=tau_j
    )

    if len(gt) == 0:
        result.tp_campaign = 0
        result.fp_campaign = len(pred)
        result.fn_campaign = 0
        result.false_campaign = len(pred) > 0
        result.unmatched_predicted_ids = [p.campaign_id for p in pred]
        result.unmatched_predicted_vehicle_count = sum(len(p.vehicles) for p in pred)
        result.campaign_precision = 0.0
        result.campaign_recall = 0.0
        result.campaign_f1 = 0.0
        # Membership: all predictions unmatched → FP_vehicle = sum |V(P)|
        tp_s, fp_s, fn_s = 0, result.unmatched_predicted_vehicle_count, 0
        result.tp_vehicle_sum = tp_s
        result.fp_vehicle_sum = fp_s
        result.fn_vehicle_sum = fn_s
        result.membership_precision_micro = _safe_div(tp_s, tp_s + fp_s)
        result.membership_recall_micro = _safe_div(tp_s, tp_s + fn_s)
        result.membership_f1_micro = _f1(
            result.membership_precision_micro, result.membership_recall_micro
        )
        result.membership_precision_including_unmatched_pred = result.membership_precision_micro
    else:
        assigned = maximum_weight_jaccard_assignment(pred, gt, tau_j=tau_j)
        matched_p = {i for i, _, _ in assigned}
        matched_g = {j for _, j, _ in assigned}

        for i, j, jacc in assigned:
            p, g = pred[i], gt[j]
            tp = len(p.vehicles & g.vehicles)
            fp = len(p.vehicles - g.vehicles)
            fn = len(g.vehicles - p.vehicles)
            p_mem = _safe_div(tp, tp + fp)
            r_mem = _safe_div(tp, tp + fn)
            result.matches.append(
                MatchPair(
                    predicted_id=p.campaign_id,
                    gt_id=g.campaign_id,
                    jaccard=jacc,
                    tp_vehicle=tp,
                    fp_vehicle=fp,
                    fn_vehicle=fn,
                    membership_precision=p_mem,
                    membership_recall=r_mem,
                    membership_f1=_f1(p_mem, r_mem),
                )
            )

        result.tp_campaign = len(assigned)
        result.fp_campaign = len(pred) - result.tp_campaign
        result.fn_campaign = len(gt) - result.tp_campaign
        result.campaign_precision = _safe_div(result.tp_campaign, result.n_predicted)
        result.campaign_recall = _safe_div(result.tp_campaign, result.n_gt)
        result.campaign_f1 = _f1(result.campaign_precision, result.campaign_recall)

        result.unmatched_predicted_ids = [
            pred[i].campaign_id for i in range(len(pred)) if i not in matched_p
        ]
        result.unmatched_gt_ids = [
            gt[j].campaign_id for j in range(len(gt)) if j not in matched_g
        ]
        result.unmatched_predicted_vehicle_count = sum(
            len(pred[i].vehicles) for i in range(len(pred)) if i not in matched_p
        )

        tp_s = sum(m.tp_vehicle for m in result.matches)
        fp_s = sum(m.fp_vehicle for m in result.matches)
        fn_s = sum(m.fn_vehicle for m in result.matches)
        for j in range(len(gt)):
            if j not in matched_g:
                fn_s += len(gt[j].vehicles)
        if include_unmatched_pred_in_primary_mem_fp:
            for i in range(len(pred)):
                if i not in matched_p:
                    fp_s += len(pred[i].vehicles)

        result.tp_vehicle_sum = tp_s
        result.fp_vehicle_sum = fp_s
        result.fn_vehicle_sum = fn_s
        result.membership_precision_micro = _safe_div(tp_s, tp_s + fp_s)
        result.membership_recall_micro = _safe_div(tp_s, tp_s + fn_s)
        result.membership_f1_micro = _f1(
            result.membership_precision_micro, result.membership_recall_micro
        )
        # V2 audit field: always the "including unmatched pred" MemP
        fp_with_unmatched = sum(m.fp_vehicle for m in result.matches) + result.unmatched_predicted_vehicle_count
        result.membership_precision_including_unmatched_pred = _safe_div(
            tp_s, tp_s + fp_with_unmatched
        )

    # Incorrect merging (independent of matcher)
    if incident_of is None:
        incident_of = {}
        for g in gt:
            for v in g.vehicles:
                incident_of[v] = g.campaign_id
    bad = incorrect_merge_flags(pred, incident_of)
    result.incorrect_merge_predicted_ids = bad
    result.incorrect_merge_run = len(bad) > 0

    # Fragmentation per GT
    frag_flags: list[int] = []
    for g in gt:
        fr = fragments_of_gt(pred, g)
        result.fragments_per_gt[g.campaign_id] = fr
        flag = 1 if fr > 1 else 0
        result.fragmentation_per_gt[g.campaign_id] = flag
        frag_flags.append(flag)
    result.fragmentation_rate_mean = float(np.mean(frag_flags)) if frag_flags else 0.0

    return result
