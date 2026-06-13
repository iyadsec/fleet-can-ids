"""Collect and align predictions from validated run directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.hierarchical_alignment.metrics import (
    compute_fleet_campaign_run_metrics,
    compute_local_event_metrics,
    compute_membership_errors,
    compute_weak_campaign_support,
)
from src.experiments.hierarchical_alignment.transform import (
    LocalThresholds,
    align_event_predictions,
    validate_local_not_overwritten,
)

PHASE3_RUNS = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")
PHASE2_ROOT = Path("new_experiments/final_validated_runs/results")
MAIN_METHODS = ("local_ids", "descriptor_clustering", "fcgnn")

SCENARIO_MAP = {
    "strong": "S3",
    "weak": "S4",
    "S0_benign_control": "S0",
    "S1_isolated": "S1",
    "S2_non_coordinated": "S2",
    "S3_strong_campaign": "S3",
    "S4_weak_campaign": "S4",
}


def _load_membership(run_dir: Path) -> pd.DataFrame:
    for name in ("vehicle_membership.csv", "scenario_membership.csv"):
        p = run_dir / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No membership in {run_dir}")


def process_run(
    run_dir: Path,
    *,
    thresholds: LocalThresholds,
) -> tuple[pd.DataFrame, dict, dict, dict, dict] | None:
    ep_path = run_dir / "event_predictions.csv"
    if not ep_path.exists():
        return None
    raw = pd.read_csv(ep_path)
    m = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
    method = str(m["method"])
    if method not in MAIN_METHODS:
        return None
    membership = _load_membership(run_dir)
    scenario_key = str(m.get("scenario_key", ""))
    if scenario_key.startswith("campaign_size_corrected"):
        scenario_id = SCENARIO_MAP.get(str(m.get("attack_strength", "")), "S?")
    else:
        scenario_id = SCENARIO_MAP.get(scenario_key, scenario_key[:2] if scenario_key else "S?")

    aligned = align_event_predictions(raw, method=method, thresholds=thresholds)
    errs = validate_local_not_overwritten(raw, aligned)
    if errs:
        raise ValueError(f"{run_dir.name}: {errs[0]}")
    framework_config = str(aligned["framework_config"].iloc[0])

    if "ground_truth_campaign_member" not in aligned.columns and "ground_truth_campaign_member" in membership.columns:
        aligned = aligned.merge(
            membership[["event_id", "ground_truth_campaign_member"]].drop_duplicates("event_id"),
            on="event_id",
            how="left",
        )
    elif "ground_truth_campaign_member" not in aligned.columns:
        gt_camp = membership.get("ground_truth_campaign_id", pd.Series("", index=membership.index)).astype(str)
        aligned["ground_truth_campaign_member"] = aligned["event_id"].map(
            membership.assign(
                _gt_member=(gt_camp.str.len() > 0).astype(int)
            ).drop_duplicates("event_id").set_index("event_id")["_gt_member"]
        ).fillna(0).astype(int)

    schema_cols = [
        "event_id",
        "scenario_vehicle_id",
        "vehicle_token",
        "local_anomaly_score",
        "local_evidence_level",
        "local_event_alert",
        "fleet_cluster_id",
        "fleet_campaign_member",
        "fleet_campaign_confidence",
        "fleet_decision",
        "ground_truth_malicious",
        "ground_truth_campaign_member",
    ]
    for col in schema_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    aligned = aligned[schema_cols]

    # Keep fleet eval columns on copy for metrics
    eval_df = raw.copy()
    eval_df["local_event_alert"] = aligned["local_event_alert"]
    eval_df["fleet_campaign_member"] = aligned["fleet_campaign_member"]
    eval_df["fleet_decision"] = aligned["fleet_decision"]
    eval_df["local_evidence_level"] = aligned["local_evidence_level"]

    seed = int(m["seed"])
    cs = int(m.get("campaign_size", 0))
    coord = float(m.get("coordination_strength", 1.0))
    expect = scenario_id in ("S3", "S4")
    latency = float(m.get("runtime_total_sec", np.nan))

    local_m = compute_local_event_metrics(
        aligned, run_id=run_dir.name, scenario_id=scenario_id, seed=seed
    )
    local_m["latency_sec"] = latency
    fleet_m = compute_fleet_campaign_run_metrics(
        eval_df,
        membership,
        run_id=run_dir.name,
        method=method,
        scenario_id=scenario_id,
        seed=seed,
        campaign_size=cs,
        coordination_strength=coord,
        expect_campaign=expect,
    )
    weak_m = compute_weak_campaign_support(
        eval_df, membership, run_id=run_dir.name, method=method, seed=seed, campaign_size=cs
    )
    err_m = compute_membership_errors(eval_df, run_id=run_dir.name, method=method)

    meta = {
        "run_id": run_dir.name,
        "method": method,
        "framework_config": framework_config,
        "scenario_id": scenario_id,
        "scenario_key": scenario_key,
        "seed": seed,
        "campaign_size": cs,
        "coordination_strength": coord,
    }
    aligned["run_id"] = run_dir.name
    aligned["seed"] = seed
    aligned["scenario_id"] = scenario_id
    aligned["framework_config"] = framework_config
    aligned["method"] = method
    return aligned, local_m, fleet_m, weak_m, {**err_m, **meta}


def collect_all_runs(
    *,
    thresholds: LocalThresholds | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = thresholds or LocalThresholds()
    aligned_rows: list[pd.DataFrame] = []
    local_rows: list[dict] = []
    fleet_rows: list[dict] = []
    weak_rows: list[dict] = []
    err_rows: list[dict] = []
    seen_local: set[tuple] = set()

    sources: list[Path] = []
    if PHASE3_RUNS.exists():
        sources.extend(sorted(PHASE3_RUNS.iterdir()))
    for sc in ("S0_benign_control", "S1_isolated", "S2_non_coordinated", "S3_strong_campaign", "S4_weak_campaign"):
        rd = PHASE2_ROOT / sc / "runs"
        if rd.exists():
            sources.extend(sorted(rd.iterdir()))

    for run_dir in sources:
        if not run_dir.is_dir():
            continue
        try:
            result = process_run(run_dir, thresholds=thresholds)
        except FileNotFoundError:
            continue
        if result is None:
            continue
        aligned, local_m, fleet_m, weak_m, err_m = result
        aligned_rows.append(aligned)
        fleet_rows.append(fleet_m)
        weak_rows.append(weak_m)
        err_rows.append(err_m)
        # Local metrics identical across methods — dedupe by scenario/seed/cs/coord
        key = (
            local_m["scenario_id"],
            local_m["seed"],
            err_m.get("campaign_size", 0),
            err_m.get("coordination_strength", 0),
        )
        if key not in seen_local:
            seen_local.add(key)
            local_rows.append(local_m)

    return (
        pd.concat(aligned_rows, ignore_index=True) if aligned_rows else pd.DataFrame(),
        pd.DataFrame(local_rows),
        pd.DataFrame(fleet_rows),
        pd.DataFrame(weak_rows),
        pd.DataFrame(err_rows),
    )
