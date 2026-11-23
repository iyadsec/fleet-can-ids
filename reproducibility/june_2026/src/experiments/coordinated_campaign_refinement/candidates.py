"""Fleet graph candidate selection and annotation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.data_splits import is_benign_attack_type


def annotate_fleet_candidates(
    df: pd.DataFrame,
    *,
    weak_threshold: float = 0.55,
    strong_threshold: float = 0.80,
) -> pd.DataFrame:
    """Tag every descriptor; evaluation universe stays complete."""
    out = df.copy()
    score = pd.to_numeric(out["anomaly_score"], errors="coerce").fillna(0.0)
    mal = ~out["attack_type"].map(is_benign_attack_type)

    out["local_evidence_level"] = np.where(
        ~mal,
        "benign",
        np.where(score >= strong_threshold, "strong", np.where(score >= weak_threshold, "weak", "below_threshold")),
    )
    out["fleet_candidate"] = score >= weak_threshold
    out["candidate_reason"] = np.where(
        out["fleet_candidate"],
        np.where(out["local_evidence_level"] == "strong", "strong_anomaly", "weak_anomaly"),
        "below_weak_threshold",
    )
    out["ground_truth_malicious"] = out.get(
        "ground_truth_malicious",
        out["attack_type"].map(lambda a: int(not is_benign_attack_type(a))),
    ).astype(int)
    return out


def candidate_statistics(df: pd.DataFrame) -> dict[str, Any]:
    cand = df[df["fleet_candidate"]]
    gt_mal = df["ground_truth_malicious"] == 1
    gt_ben = ~gt_mal
    cand_mal = cand["ground_truth_malicious"] == 1
    cand_ben = cand["ground_truth_malicious"] == 0
    tp = int(cand_mal.sum())
    fn = int((gt_mal & ~df["fleet_candidate"]).sum())
    fp = int(cand_ben.sum())
    tn = int((gt_ben & ~df["fleet_candidate"]).sum())
    return {
        "total_descriptors": len(df),
        "candidate_nodes": len(cand),
        "malicious_candidates": int(cand_mal.sum()),
        "benign_candidates": int(cand_ben.sum()),
        "weak_malicious_candidates": int((cand["local_evidence_level"] == "weak").sum()),
        "strong_malicious_candidates": int((cand["local_evidence_level"] == "strong").sum()),
        "benign_below_threshold": int((gt_ben & ~df["fleet_candidate"]).sum()),
        "candidate_precision": tp / max(tp + fp, 1),
        "candidate_recall": tp / max(tp + fn, 1),
        "correctly_filtered_benign": tn,
        "benign_incorrectly_promoted": fp,
    }
