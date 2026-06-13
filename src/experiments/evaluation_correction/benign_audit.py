"""Audit benign descriptors on attacked vehicle instances."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type
from src.experiments.evaluation_correction.promotion import PromotionConfig, classify_local_evidence_row


def audit_benign_on_attacked(
    run_dir: Path,
    *,
    cfg: PromotionConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or PromotionConfig()
    events = pd.read_csv(run_dir / "event_predictions.csv")
    scenario = pd.read_csv(run_dir / "selected_source_records.csv")
    metrics = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]

    merged = events.merge(
        scenario[
            [
                c
                for c in (
                    "event_id",
                    "source_trace",
                    "source_segment",
                    "attack_type",
                    "ground_truth_malicious",
                    "scenario_role",
                )
                if c in scenario.columns
            ]
        ],
        on="event_id",
        how="left",
        suffixes=("", "_src"),
    )
    if "ground_truth_malicious_src" in merged.columns:
        merged["window_ground_truth_malicious"] = merged["ground_truth_malicious_src"].fillna(
            merged["ground_truth_malicious"]
        )
    else:
        merged["window_ground_truth_malicious"] = merged["ground_truth_malicious"]

    attacked = merged[merged["scenario_role"] == "coordinated"].copy()
    benign_sel = attacked[
        (attacked["window_ground_truth_malicious"] == 0)
        | attacked["attack_type"].map(is_benign_attack_type)
    ].copy()

    rows: list[dict] = []
    for _, row in benign_sel.iterrows():
        atk = str(row.get("attack_type", ""))
        gt_mal = int(row.get("window_ground_truth_malicious", 0))
        evidence = classify_local_evidence_row(
            row, weak_threshold=cfg.weak_threshold, strong_threshold=cfg.strong_threshold
        )
        selected = int(row.get("ground_truth_malicious", 1) == 0)
        if gt_mal == 1:
            reason = "FAIL: ground_truth_malicious=1 relabelled benign"
            passed = False
        elif not is_benign_attack_type(atk) and evidence != "benign":
            reason = f"benign-on-attacked from {atk} segment with {evidence}"
            passed = True
        elif is_benign_attack_type(atk):
            reason = "true benign attack_type window"
            passed = True
        else:
            reason = "benign ground_truth window on attacked vehicle"
            passed = True

        rows.append(
            {
                "run_id": run_dir.name,
                "seed": int(metrics.get("seed", -1)),
                "scenario_vehicle_id": row.get("scenario_vehicle_id", row.get("vehicle_token", "")),
                "source_trace": row.get("source_trace", ""),
                "source_segment": row.get("source_segment", ""),
                "event_id": row["event_id"],
                "original_attack_label": atk,
                "window_ground_truth_malicious": gt_mal,
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "local_evidence_level": evidence,
                "selected_as_benign": selected,
                "selection_reason": reason,
                "validation_passed": passed and gt_mal == 0,
            }
        )
    return pd.DataFrame(rows)
