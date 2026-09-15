"""Campaign and local metrics for the controlled reviewer ablation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def match_campaigns(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    min_jaccard: float = 0.5,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Greedy Jaccard matching of predicted clusters to GT campaigns (noise=-1 ignored)."""
    gt_ids = sorted(set(gt.tolist()) - {-1})
    pred_ids = sorted(set(pred.tolist()) - {-1})
    gt_sets = {cid: set(np.where(gt == cid)[0].tolist()) for cid in gt_ids}
    pred_sets = {cid: set(np.where(pred == cid)[0].tolist()) for cid in pred_ids}

    pairs: List[Tuple[float, int, int]] = []
    for g in gt_ids:
        for p in pred_ids:
            pairs.append((_jaccard(gt_sets[g], pred_sets[p]), g, p))
    pairs.sort(reverse=True)

    matched: Dict[int, int] = {}
    used_gt, used_pred = set(), set()
    for jac, g, p in pairs:
        if jac < min_jaccard:
            break
        if g in used_gt or p in used_pred:
            continue
        matched[g] = p
        used_gt.add(g)
        used_pred.add(p)

    unmatched_gt = [g for g in gt_ids if g not in used_gt]
    unmatched_pred = [p for p in pred_ids if p not in used_pred]
    return matched, unmatched_gt, unmatched_pred


def campaign_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    vehicles: Sequence[str],
    *,
    is_unrelated: bool = False,
) -> Dict[str, Any]:
    """
    Matching: greedy one-to-one Jaccard ≥ 0.5 between predicted and GT campaign node sets.
    Campaign Precision = matched / #predicted campaigns
    Campaign Recall    = matched / #GT campaigns
    Campaign F1        = harmonic mean of precision and recall
    Membership F1      = micro F1 over node membership of matched pairs
      (unmatched GT nodes → FN; unmatched pred nodes → FP)
    Fragmentation      = mean over GT campaigns of (#pred clusters with any overlap)
    Incorrect merge rate: fraction of predicted campaigns spanning ≥2 GT campaign IDs
    False campaign rate: fraction of predicted campaigns with no GT match
    """
    del vehicles  # reserved for vehicle-level extensions; node metrics use indices
    gt = np.asarray(gt, dtype=int)
    pred = np.asarray(pred, dtype=int)

    gt_ids = sorted(set(gt.tolist()) - {-1})
    pred_ids = sorted(set(pred.tolist()) - {-1})
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)

    matched, unmatched_gt, unmatched_pred = match_campaigns(gt, pred)
    n_matched = len(matched)

    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    tp = fp = fn = 0
    for g, p in matched.items():
        gset = set(np.where(gt == g)[0].tolist())
        pset = set(np.where(pred == p)[0].tolist())
        tp += len(gset & pset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    for g in unmatched_gt:
        fn += int((gt == g).sum())
    for p in unmatched_pred:
        fp += int((pred == p).sum())
    if tp + fp + fn == 0:
        mem_f1 = 0.0
    else:
        mem_p = tp / (tp + fp) if (tp + fp) else 0.0
        mem_r = tp / (tp + fn) if (tp + fn) else 0.0
        mem_f1 = 2 * mem_p * mem_r / (mem_p + mem_r) if (mem_p + mem_r) else 0.0

    frag_vals = []
    for g in gt_ids:
        gset = set(np.where(gt == g)[0].tolist())
        overlaps = 0
        for p in pred_ids:
            pset = set(np.where(pred == p)[0].tolist())
            if gset & pset:
                overlaps += 1
        frag_vals.append(float(overlaps))
    fragmentation = float(np.mean(frag_vals)) if frag_vals else 0.0

    merge_count = 0
    for p in pred_ids:
        idxs = np.where(pred == p)[0]
        gt_in = {int(gt[i]) for i in idxs if int(gt[i]) != -1}
        if len(gt_in) >= 2:
            merge_count += 1
    incorrect_merge_rate = merge_count / n_pred if n_pred else 0.0
    false_campaign_rate = len(unmatched_pred) / n_pred if n_pred else 0.0

    return {
        "campaign_precision": float(precision),
        "campaign_recall": float(recall),
        "campaign_f1": float(f1),
        "membership_f1": float(mem_f1),
        "fragmentation": float(fragmentation),
        "incorrect_merge_rate": float(incorrect_merge_rate),
        "false_campaign_rate": float(false_campaign_rate),
        "n_gt_campaigns": int(n_gt),
        "n_pred_campaigns": int(n_pred),
        "n_matched_campaigns": int(n_matched),
        "is_unrelated": bool(is_unrelated),
    }


def local_detection_metrics(
    y_true_attack: np.ndarray,
    local_alert: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y_true_attack, dtype=bool)
    p = np.asarray(local_alert, dtype=bool)
    tp = int(((y) & (p)).sum())
    fp = int(((~y) & (p)).sum())
    fn = int(((y) & (~p)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "local_precision": float(prec),
        "local_recall": float(rec),
        "local_f1": float(f1),
    }
