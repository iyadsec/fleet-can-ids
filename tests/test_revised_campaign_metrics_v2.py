"""Unit tests for revised_jaccard_0.5_v3 — synthetic vehicle-id sets only.

Retains the V2 scenario coverage; membership assertions follow V3 primary micro
(unmatched predicted campaigns contribute to membership FP).
"""

from __future__ import annotations

import pytest

from src.evaluation.revised_campaign_metrics_v2 import (
    TAU_J,
    GroundTruthCampaign,
    PredictedCampaign,
    evaluate_protocol_v2,
    evaluate_protocol_v3,
    jaccard,
    maximum_weight_jaccard_assignment,
    predicted_campaigns_from_clusters,
)


def _P(cid, *vehicles) -> PredictedCampaign:
    return PredictedCampaign(campaign_id=cid, vehicles=frozenset(vehicles))


def _G(cid, *vehicles) -> GroundTruthCampaign:
    return GroundTruthCampaign(campaign_id=cid, vehicles=frozenset(vehicles))


def test_tau_j_frozen_not_tuned():
    assert TAU_J == 0.5


def test_1_perfect_one_campaign_match():
    pred = [_P("p0", "A", "B", "C")]
    gt = [_G("g0", "A", "B", "C")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 0
    assert r.fn_campaign == 0
    assert r.campaign_precision == 1.0
    assert r.campaign_recall == 1.0
    assert r.campaign_f1 == 1.0
    assert len(r.matches) == 1
    assert r.matches[0].jaccard == 1.0
    assert r.membership_precision_micro == 1.0
    assert r.membership_recall_micro == 1.0
    assert r.membership_f1_micro == 1.0
    assert r.fragments_per_gt["g0"] == 1
    assert r.fragmentation_per_gt["g0"] == 0
    assert r.incorrect_merge_run is False


def test_2_partial_match_jaccard_ge_0_5():
    pred = [_P("p0", "A", "B", "X")]
    gt = [_G("g0", "A", "B", "C")]
    assert jaccard(pred[0].vehicles, gt[0].vehicles) == 0.5
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 0
    assert r.fn_campaign == 0
    assert r.matches[0].tp_vehicle == 2
    assert r.matches[0].fp_vehicle == 1
    assert r.matches[0].fn_vehicle == 1
    assert r.membership_precision_micro == pytest.approx(2 / 3)
    assert r.membership_recall_micro == pytest.approx(2 / 3)


def test_3_partial_match_jaccard_lt_0_5():
    pred = [_P("p0", "A", "X", "Y", "Z")]
    gt = [_G("g0", "A", "B")]
    assert jaccard(pred[0].vehicles, gt[0].vehicles) < 0.5
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 0
    assert r.fp_campaign == 1
    assert r.fn_campaign == 1
    assert r.matches == []
    # Unmatched pred → FP_vehicle=4; unmatched GT → FN_vehicle=2
    assert r.tp_vehicle_sum == 0
    assert r.fp_vehicle_sum == 4
    assert r.fn_vehicle_sum == 2
    assert r.membership_precision_micro == 0.0
    assert r.membership_recall_micro == 0.0
    assert r.membership_f1_micro == 0.0
    assert "p0" in r.unmatched_predicted_ids
    assert "g0" in r.unmatched_gt_ids


def test_4_complete_campaign_miss():
    pred: list[PredictedCampaign] = []
    gt = [_G("g0", "A", "B")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 0
    assert r.fp_campaign == 0
    assert r.fn_campaign == 1
    assert r.campaign_recall == 0.0
    assert r.membership_recall_micro == 0.0
    assert r.membership_f1_micro == 0.0
    assert r.fn_vehicle_sum == 2
    assert r.fragments_per_gt["g0"] == 0
    assert r.fragmentation_per_gt["g0"] == 0


def test_5_false_predicted_campaign():
    pred = [_P("p0", "A", "B")]
    gt: list[GroundTruthCampaign] = []
    r = evaluate_protocol_v3(pred, gt)
    assert r.n_gt == 0
    assert r.false_campaign is True
    assert r.tp_campaign == 0
    assert r.fp_campaign == 1
    assert r.fn_campaign == 0
    assert r.unmatched_predicted_vehicle_count == 2
    assert r.fp_vehicle_sum == 2
    assert r.membership_precision_micro == 0.0


def test_6_one_gt_split_across_two_predicted():
    pred = [_P("p0", "A", "B"), _P("p1", "C", "D")]
    gt = [_G("g0", "A", "B", "C", "D")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 1
    assert r.fn_campaign == 0
    assert r.fragments_per_gt["g0"] == 2
    assert r.fragmentation_per_gt["g0"] == 1
    assert len(r.unmatched_predicted_ids) == 1
    # Matched half: TP=2 FP=0 FN=2; unmatched half: FP+=2 → TP=2 FP=2 FN=2
    assert r.tp_vehicle_sum == 2
    assert r.fp_vehicle_sum == 2
    assert r.fn_vehicle_sum == 2


def test_7_two_independent_incidents_merged():
    pred = [_P("p0", "A", "B", "C", "D")]
    gt = [_G("inc1", "A", "B"), _G("inc2", "C", "D")]
    incident_of = {"A": "inc1", "B": "inc1", "C": "inc2", "D": "inc2"}
    r = evaluate_protocol_v3(pred, gt, incident_of=incident_of)
    assert r.incorrect_merge_run is True
    assert "p0" in r.incorrect_merge_predicted_ids
    assert r.tp_campaign == 1
    assert r.fp_campaign == 0
    assert r.fn_campaign == 1


def test_8_two_gt_two_correctly_separated_predictions():
    pred = [_P("p0", "A", "B"), _P("p1", "C", "D")]
    gt = [_G("g0", "A", "B"), _G("g1", "C", "D")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 2
    assert r.fp_campaign == 0
    assert r.fn_campaign == 0
    assert r.campaign_f1 == 1.0
    assert r.incorrect_merge_run is False
    assert r.fragments_per_gt["g0"] == 1
    assert r.fragments_per_gt["g1"] == 1
    assert r.fragmentation_per_gt["g0"] == 0
    assert r.fragmentation_per_gt["g1"] == 0
    assert r.membership_f1_micro == 1.0


def test_9_benign_vehicles_in_correct_prediction():
    pred = [_P("p0", "A", "B", "X", "Y")]
    gt = [_G("g0", "A", "B")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.matches[0].tp_vehicle == 2
    assert r.matches[0].fp_vehicle == 2
    assert r.matches[0].fn_vehicle == 0
    assert r.membership_precision_micro == pytest.approx(0.5)
    assert r.membership_recall_micro == 1.0
    assert r.incorrect_merge_run is False


def test_10_dbscan_noise_excluded():
    clusters = {
        0: ["A", "B"],
        -1: ["N1", "N2", "N3"],
    }
    pred = predicted_campaigns_from_clusters(clusters, noise_label=-1)
    assert len(pred) == 1
    assert pred[0].campaign_id == 0
    assert "N1" not in pred[0].vehicles
    gt = [_G("g0", "A", "B")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.n_predicted == 1
    assert r.tp_campaign == 1
    assert r.fp_campaign == 0


def test_max_weight_prefers_global_optimum_over_greedy():
    pred = [_P("p0", "A", "B", "C", "X"), _P("p1", "D", "E", "F", "Y")]
    gt = [_G("g0", "A", "B", "C"), _G("g1", "D", "E", "F")]
    pairs = maximum_weight_jaccard_assignment(pred, gt)
    assert len(pairs) == 2
    matched_gt = {j for _, j, _ in pairs}
    matched_p = {i for i, _, _ in pairs}
    assert matched_gt == {0, 1}
    assert matched_p == {0, 1}


def test_unrelated_false_campaign_counts_as_membership_fp_v3():
    pred = [_P("p_true", "A", "B"), _P("p_false", "X", "Y")]
    gt = [_G("g0", "A", "B")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 1
    assert r.fragments_per_gt["g0"] == 1
    assert r.fragmentation_per_gt["g0"] == 0
    assert "p_false" in r.unmatched_predicted_ids
    assert r.unmatched_predicted_vehicle_count == 2
    assert r.tp_vehicle_sum == 2
    assert r.fp_vehicle_sum == 2
    assert r.fn_vehicle_sum == 0
    assert r.membership_precision_micro == pytest.approx(0.5)
    assert r.membership_recall_micro == 1.0
    assert r.membership_f1_micro == pytest.approx(2 / 3)


def test_v3_matched_plus_unmatched_prediction_membership():
    """GT={A,B,C}, matched={A,B,C}, unmatched={D,E,F} → MemP=0.5 MemR=1 MemF1=2/3."""
    pred = [_P("p_match", "A", "B", "C"), _P("p_extra", "D", "E", "F")]
    gt = [_G("g0", "A", "B", "C")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 1
    assert r.tp_vehicle_sum == 3
    assert r.fp_vehicle_sum == 3
    assert r.fn_vehicle_sum == 0
    assert r.membership_precision_micro == pytest.approx(0.5)
    assert r.membership_recall_micro == 1.0
    assert r.membership_f1_micro == pytest.approx(2 / 3)


def test_v3_unmatched_gt_contributes_all_vehicles_as_fn():
    pred = [_P("p0", "A", "B")]
    gt = [_G("g0", "A", "B"), _G("g_miss", "X", "Y", "Z")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fn_campaign == 1
    assert "g_miss" in r.unmatched_gt_ids
    assert r.tp_vehicle_sum == 2
    assert r.fp_vehicle_sum == 0
    assert r.fn_vehicle_sum == 3
    assert r.membership_recall_micro == pytest.approx(2 / 5)


def test_v3_vehicle_in_two_unmatched_predictions_counts_twice():
    pred = [
        _P("p_match", "A", "B"),
        _P("p_u1", "V", "X"),
        _P("p_u2", "V", "Y"),
    ]
    gt = [_G("g0", "A", "B")]
    r = evaluate_protocol_v3(pred, gt)
    assert r.tp_campaign == 1
    assert r.fp_campaign == 2
    assert r.tp_vehicle_sum == 2
    # FP from unmatched: |{V,X}| + |{V,Y}| = 4 (V counted twice)
    assert r.fp_vehicle_sum == 4
    assert r.unmatched_predicted_vehicle_count == 4
    assert r.membership_precision_micro == pytest.approx(2 / 6)


def test_v2_archived_excludes_unmatched_pred_from_primary_memp():
    """Archived V2 semantics still available for comparison."""
    pred = [_P("p_true", "A", "B"), _P("p_false", "X", "Y")]
    gt = [_G("g0", "A", "B")]
    r2 = evaluate_protocol_v2(pred, gt)
    r3 = evaluate_protocol_v3(pred, gt)
    assert r2.membership_precision_micro == 1.0
    assert r2.membership_precision_including_unmatched_pred == pytest.approx(0.5)
    assert r3.membership_precision_micro == pytest.approx(0.5)
