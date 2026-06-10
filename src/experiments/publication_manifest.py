"""Scan, validate, and deduplicate scenario experiment runs for publication."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.scenario_registry import SCENARIO_REGISTRY, get_scenario

REQUIRED_RUN_FILES = (
    "config_snapshot.yaml",
    "run_level_metrics.csv",
    "event_predictions.csv",
    "vehicle_predictions.csv",
    "campaign_predictions.csv",
    "scenario_membership.csv",
    "selected_source_records.csv",
)

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
METHODS = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    scenario_key: str
    method: str
    seed: int
    campaign_size: int
    coordination_strength: float
    metrics: dict[str, Any]
    scenario_hash: str
    excluded: bool = False
    exclude_reason: str = ""


def _parse_run_id(run_id: str) -> dict[str, Any] | None:
    m = re.search(r"seed(\d+)", run_id)
    if not m:
        return None
    seed = int(m.group(1))
    cs = 0.0
    mcs = re.search(r"cs([\d]+)p([\d]+)", run_id)
    if mcs:
        cs = float(f"{mcs.group(1)}.{mcs.group(2)}")
    n = 0
    mn = re.search(r"_n(\d+)_", run_id)
    if mn:
        n = int(mn.group(1))
    method = None
    for meth in METHODS:
        if f"_{meth}_" in run_id or run_id.startswith(f"{run_id.split('_')[0]}_{meth}"):
            pass
    for meth in METHODS:
        if re.search(rf"_{meth}_", run_id):
            method = meth
            break
    return {"seed": seed, "coordination_strength": cs, "campaign_size": n, "method": method}


def _scenario_hash(membership_path: Path) -> str:
    if not membership_path.exists():
        return ""
    df = pd.read_csv(membership_path, usecols=["event_id"])
    payload = "|".join(sorted(df["event_id"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validate_scenario_semantics(
    scenario_key: str,
    membership: pd.DataFrame,
    event_pred: pd.DataFrame,
    metrics: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    spec = get_scenario(scenario_key)
    gt_campaigns = membership.loc[
        membership["ground_truth_campaign_id"].astype(str).str.len() > 0, "ground_truth_campaign_id"
    ].nunique()
    attacked_vehicles = membership.loc[membership["ground_truth_malicious"] == 1, "vehicle_model"].nunique()
    mal_events = int(membership["ground_truth_malicious"].sum())

    if spec.scenario_id == "S0":
        if mal_events > 0:
            errors.append(f"S0 has {mal_events} malicious events")
        if gt_campaigns > 0:
            errors.append("S0 has ground-truth campaigns")
    elif spec.scenario_id == "S1":
        if attacked_vehicles != 1:
            errors.append(f"S1 has {attacked_vehicles} attacked vehicles (expected 1)")
        if gt_campaigns > 0:
            errors.append("S1 has ground-truth campaign")
    elif spec.scenario_id == "S2":
        if attacked_vehicles < 2:
            errors.append(f"S2 has only {attacked_vehicles} attacked vehicles")
    elif spec.scenario_id in ("S3", "S4"):
        if attacked_vehicles < 2:
            errors.append(f"{spec.scenario_id} has only {attacked_vehicles} attacked vehicles")
        if spec.scenario_id == "S4" and mal_events > 0:
            weak_th, strong_th = 0.55, 0.80
            mal = membership[membership["ground_truth_malicious"] == 1]
            if "anomaly_score" in mal.columns:
                out_of_band = mal[
                    (mal["anomaly_score"] < weak_th) | (mal["anomaly_score"] >= strong_th)
                ]
                if len(out_of_band) > len(mal) * 0.5:
                    errors.append("S4 majority of malicious events outside weak band")

    if "fpr" in metrics and not pd.isna(metrics.get("fpr")):
        if "ground_truth_malicious" in event_pred.columns:
            y_true = event_pred["ground_truth_malicious"].astype(int)
            if (y_true == 0).sum() == 0:
                errors.append("FPR computed with no negative examples")

    qualifying = event_pred.get("vehicles_in_cluster", pd.Series(dtype=float))
    if len(qualifying) and (qualifying == 1).any() and (event_pred.get("final_decision") == "coordinated_attack").any():
        one_veh_coord = event_pred[
            (event_pred.get("final_decision") == "coordinated_attack")
            & (event_pred.get("vehicles_in_cluster", 0) < 2)
        ]
        if len(one_veh_coord):
            errors.append("coordinated decision with <2 vehicles in cluster")

    return errors


def scan_runs(results_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for mpath in results_root.rglob("runs/*/run_level_metrics.csv"):
        run_dir = mpath.parent
        run_id = run_dir.name
        if not all((run_dir / f).exists() for f in REQUIRED_RUN_FILES):
            continue
        metrics = pd.read_csv(mpath).iloc[0].to_dict()
        scenario_key = str(metrics.get("scenario_key", ""))
        if scenario_key not in SCENARIO_REGISTRY:
            parent = run_dir.parent.parent.name
            scenario_key = parent
        membership = pd.read_csv(run_dir / "scenario_membership.csv")
        event_pred = pd.read_csv(run_dir / "event_predictions.csv")
        sem_errs = _validate_scenario_semantics(scenario_key, membership, event_pred, metrics)
        rec = RunRecord(
            run_id=run_id,
            run_dir=run_dir,
            scenario_key=scenario_key,
            method=str(metrics.get("method", "")),
            seed=int(metrics.get("seed", -1)),
            campaign_size=int(metrics.get("campaign_size", 0)),
            coordination_strength=float(metrics.get("coordination_strength", 0.0)),
            metrics=metrics,
            scenario_hash=_scenario_hash(run_dir / "scenario_membership.csv"),
            excluded=bool(sem_errs),
            exclude_reason="; ".join(sem_errs),
        )
        records.append(rec)
    return records


def deduplicate_runs(records: list[RunRecord]) -> tuple[list[RunRecord], pd.DataFrame]:
    """Keep latest valid run per (scenario, method, seed, n, cs)."""
    excluded_rows: list[dict] = []
    valid = [r for r in records if not r.excluded]
    for r in records:
        if r.excluded:
            excluded_rows.append({"run_id": r.run_id, "reason": r.exclude_reason})

    df = pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "scenario_key": r.scenario_key,
                "method": r.method,
                "seed": r.seed,
                "campaign_size": r.campaign_size,
                "coordination_strength": r.coordination_strength,
                "run_dir": str(r.run_dir),
                "scenario_hash": r.scenario_hash,
                "timestamp": r.run_id.split("_")[-2] if "_" in r.run_id else "",
            }
            for r in valid
        ]
    )
    if df.empty:
        return [], pd.DataFrame(excluded_rows)

    df = df.sort_values("timestamp").drop_duplicates(
        ["scenario_key", "method", "seed", "campaign_size", "coordination_strength"],
        keep="last",
    )
    kept_ids = set(df["run_id"])
    kept = [r for r in valid if r.run_id in kept_ids]
    return kept, pd.DataFrame(excluded_rows)


def expected_combinations(config: dict[str, Any]) -> pd.DataFrame:
    from src.experiments.scenario_registry import enumerate_run_plan

    rows = enumerate_run_plan(
        scenario_keys=list(SCENARIO_REGISTRY.keys()),
        methods=METHODS,
        seeds=list(config.get("general", {}).get("seeds", REQUIRED_SEEDS)),
        campaign_sizes=list(config.get("campaign", {}).get("campaign_sizes", [2, 5, 10])),
        coordination_strengths=list(config.get("campaign", {}).get("coordination_strengths", [0.0, 0.25, 0.5, 0.75, 1.0])),
    )
    return pd.DataFrame(rows)


def missing_combinations(validated: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    if validated.empty:
        return expected.copy()
    keys = ["scenario_key", "method", "seed", "campaign_size", "coordination_strength"]
    merged = expected.merge(validated[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])


def build_validated_manifest(
    results_root: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[RunRecord]]:
    records = scan_runs(results_root)
    kept, excluded = deduplicate_runs(records)
    expected = expected_combinations(config)

    validated_rows = []
    for r in kept:
        row = {
            "run_id": r.run_id,
            "run_dir": str(r.run_dir),
            "scenario_key": r.scenario_key,
            "scenario_id": get_scenario(r.scenario_key).scenario_id,
            "method": r.method,
            "seed": r.seed,
            "campaign_size": r.campaign_size,
            "coordination_strength": r.coordination_strength,
            "scenario_hash": r.scenario_hash,
            **{k: r.metrics.get(k) for k in r.metrics},
        }
        validated_rows.append(row)
    validated = pd.DataFrame(validated_rows)
    missing = missing_combinations(
        validated[["scenario_key", "method", "seed", "campaign_size", "coordination_strength"]]
        if not validated.empty
        else pd.DataFrame(columns=["scenario_key", "method", "seed", "campaign_size", "coordination_strength"]),
        expected,
    )
    return validated, excluded, missing, kept
