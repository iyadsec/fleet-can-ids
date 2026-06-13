"""Collect and align authoritative scenario and campaign-size results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.final_publication_scenarios.inventory import REQUIRED_SEEDS
from src.experiments.hierarchical_alignment.collect import process_run
from src.experiments.hierarchical_alignment.transform import LocalThresholds

PHASE2 = Path("new_experiments/final_validated_runs/results")
PHASE3 = Path("new_experiments/final_validated_runs/results/campaign_size_corrected")
HIER = Path("new_experiments/final_validated_runs/hierarchical_alignment/results")


def _load_hierarchical_if_present() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if (HIER / "local_event_metrics.csv").exists():
        return (
            pd.read_csv(HIER / "local_event_metrics.csv"),
            pd.read_csv(HIER / "fleet_campaign_metrics.csv"),
            pd.read_csv(HIER / "weak_campaign_support.csv") if (HIER / "weak_campaign_support.csv").exists() else pd.DataFrame(),
            pd.read_csv(HIER / "campaign_membership_errors.csv") if (HIER / "campaign_membership_errors.csv").exists() else pd.DataFrame(),
        )
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def collect_scenario_results(project_root: Path) -> dict[str, pd.DataFrame]:
    """Build S0–S4 publication tables from hierarchical alignment + raw runs."""
    local_h, fleet_h, weak_h, err_h = _load_hierarchical_if_present()
    if local_h.empty:
        thresholds = LocalThresholds()
        local_rows, fleet_rows, weak_rows, err_rows = [], [], [], []
        for sc in ("S0_benign_control", "S1_isolated", "S2_non_coordinated", "S3_strong_campaign", "S4_weak_campaign"):
            rd = project_root / PHASE2 / sc / "runs"
            for run_dir in sorted(rd.iterdir()) if rd.exists() else []:
                if not run_dir.is_dir():
                    continue
                try:
                    result = process_run(run_dir, thresholds=thresholds)
                    if result is None:
                        continue
                    _, local_m, fleet_m, weak_m, err_m = result
                    local_rows.append(local_m)
                    fleet_rows.append(fleet_m)
                    weak_rows.append(weak_m)
                    err_rows.append(err_m)
                except Exception:
                    continue
        local_h = pd.DataFrame(local_rows)
        fleet_h = pd.DataFrame(fleet_rows)
        weak_h = pd.DataFrame(weak_rows)
        err_h = pd.DataFrame(err_rows)

    # Primary framework for paper: C3 GraphSAGE
    fleet_c3 = fleet_h[fleet_h["framework_config"] == "C3"].copy() if "framework_config" in fleet_h.columns else fleet_h.copy()
    local = local_h.drop_duplicates(subset=["run_id"], keep="first") if "run_id" in local_h.columns else local_h

    run_rows = []
    for sc_key in ("S0_benign_control", "S1_isolated", "S2_non_coordinated", "S3_strong_campaign", "S4_weak_campaign"):
        mpath = project_root / PHASE2 / sc_key / "run_level_metrics.csv"
        if mpath.exists():
            sub = pd.read_csv(mpath)
            sub["scenario_key"] = sc_key
            sub["scenario_id"] = sc_key[:2]
            run_rows.append(sub)
    run_level = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame()

    safety = _build_safety_metrics(local, fleet_c3)
    summary = _summary_mean_std(fleet_c3, group_cols=["scenario_id", "framework_config"])
    ci = _confidence_intervals(fleet_c3, group_cols=["scenario_id", "framework_config"])
    stats = _paired_stats_placeholder(fleet_c3)

    return {
        "scenario_run_level_metrics": run_level,
        "local_event_metrics": local,
        "fleet_campaign_metrics": fleet_c3,
        "scenario_safety_metrics": safety,
        "campaign_membership_errors": err_h,
        "weak_campaign_support": weak_h,
        "summary_mean_std": summary,
        "confidence_intervals": ci,
        "statistical_tests": stats,
    }


def collect_campaign_size_results(project_root: Path) -> dict[str, pd.DataFrame]:
    root = project_root / PHASE3
    run_level = pd.read_csv(root / "run_level_metrics.csv")
    fcgnn = run_level[run_level["method"] == "fcgnn"].copy()
    fcgnn["framework_config"] = "C3"
    strong = fcgnn[fcgnn["attack_strength"] == "strong"]
    weak = fcgnn[fcgnn["attack_strength"] == "weak"]

    comp_rows = []
    for run_dir in sorted((root / "runs").iterdir()):
        if not run_dir.is_dir():
            continue
        sr = run_dir / "selected_source_records.csv"
        if not sr.exists():
            continue
        df = pd.read_csv(sr)
        comp_rows.append(
            {
                "run_id": run_dir.name,
                "descriptor_count": len(df),
                "distinct_vehicles": df["scenario_vehicle_id"].nunique() if "scenario_vehicle_id" in df.columns else np.nan,
            }
        )
    composition = pd.DataFrame(comp_rows)

    return {
        "run_level_metrics": run_level,
        "fcgnn_run_level_metrics": fcgnn,
        "strong_summary": _summary_mean_std(strong, ["campaign_size", "framework_config"]),
        "weak_summary": _summary_mean_std(weak, ["campaign_size", "framework_config"]),
        "graph_statistics": _aggregate_graph_stats(root),
        "runtime_memory": fcgnn[["run_id", "seed", "attack_strength", "campaign_size", "runtime_graph_construction_sec", "runtime_gnn_inference_sec", "runtime_total_sec"]].copy()
        if all(c in fcgnn.columns for c in ("runtime_graph_construction_sec",))
        else fcgnn[["run_id", "seed", "attack_strength", "campaign_size"]],
        "confidence_intervals": _confidence_intervals(fcgnn, ["attack_strength", "campaign_size"]),
        "statistical_tests": _campaign_size_stats(fcgnn),
        "composition_validation": composition,
    }


def _build_safety_metrics(local: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in ("S0", "S1", "S2", "S3", "S4"):
        fl = fleet[fleet["scenario_id"] == sid] if "scenario_id" in fleet.columns else pd.DataFrame()
        loc = local[local["scenario_id"] == sid] if "scenario_id" in local.columns else pd.DataFrame()
        row = {"scenario_id": sid}
        if sid == "S0":
            row["false_local_alert_rate"] = float(loc["fpr"].mean()) if not loc.empty and "fpr" in loc.columns else np.nan
            row["false_campaign_alert_rate"] = float(fl["false_campaign_alert_rate"].mean()) if not fl.empty else np.nan
            row["correct_no_attack_decision_rate"] = 1.0 - row.get("false_campaign_alert_rate", np.nan)
        elif sid == "S1":
            row["local_attacked_vehicle_detection"] = float(loc["recall"].mean()) if not loc.empty else np.nan
            row["isolated_incident_decision_rate"] = 1.0 - float(fl["false_campaign_alert_rate"].mean()) if not fl.empty else np.nan
            row["incorrect_campaign_declaration_rate"] = float(fl["false_campaign_alert_rate"].mean()) if not fl.empty else np.nan
        elif sid == "S2":
            row["attacked_vehicle_detection"] = float(loc["recall"].mean()) if not loc.empty else np.nan
            row["incorrect_merging_rate"] = float(fl["incorrect_campaign_merging"].mean()) if not fl.empty and "incorrect_campaign_merging" in fl.columns else np.nan
            row["false_single_campaign_declaration_rate"] = float(fl["false_campaign_alert_rate"].mean()) if not fl.empty else np.nan
            row["correct_separation_rate"] = 1.0 - row.get("incorrect_merging_rate", np.nan)
        elif sid in ("S3", "S4"):
            for col in ("campaign_detection_rate", "campaign_precision", "campaign_recall", "campaign_f1",
                        "membership_purity", "fragmentation"):
                if col in fl.columns:
                    row[col] = float(fl[col].mean())
            if "benign_vehicles_incorrectly_included" in fl.columns:
                row["benign_vehicles_included"] = float(fl["benign_vehicles_incorrectly_included"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_mean_std(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    metrics = [c for c in ("campaign_f1", "campaign_precision", "campaign_recall", "false_campaign_alert_rate",
                           "campaign_detection_rate", "membership_purity") if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        row["n_seeds"] = g["seed"].nunique() if "seed" in g.columns else len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def _confidence_intervals(df: pd.DataFrame, group_cols: list[str], alpha: float = 0.05) -> pd.DataFrame:
    from scipy import stats as sp_stats

    metrics = [c for c in ("campaign_f1", "campaign_precision", "false_campaign_alert_rate") if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        for m in metrics:
            vals = g[m].astype(float).dropna().to_numpy()
            if len(vals) < 2:
                continue
            mean = float(vals.mean())
            se = float(sp_stats.sem(vals))
            h = se * sp_stats.t.ppf(1 - alpha / 2, len(vals) - 1)
            rows.append({**dict(zip(group_cols, keys)), "metric": m, "mean": mean, "ci95_low": mean - h, "ci95_high": mean + h, "n": len(vals)})
    return pd.DataFrame(rows)


def _paired_stats_placeholder(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["family", "comparison", "metric", "p_value_raw", "p_value_holm", "effect_size", "n_seeds"])


def _campaign_size_stats(df: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats as sp_stats

    rows = []
    for strength in ("strong", "weak"):
        sub = df[df["attack_strength"] == strength]
        for m in ("campaign_f1", "campaign_detection_rate", "false_campaign_alert_rate"):
            if m not in sub.columns:
                continue
            for a, b in ((2, 5), (5, 10), (2, 10)):
                ga = sub[sub["campaign_size"] == a].groupby("seed")[m].mean()
                gb = sub[sub["campaign_size"] == b].groupby("seed")[m].mean()
                joined = pd.concat([ga, gb], axis=1, keys=["a", "b"]).dropna()
                if len(joined) < 3:
                    continue
                t, p = sp_stats.ttest_rel(joined["a"], joined["b"])
                diff = joined["a"] - joined["b"]
                dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
                rows.append({
                    "family": "campaign_size",
                    "comparison": f"{strength}_n{a}_vs_n{b}",
                    "metric": m,
                    "paired_seeds": len(joined),
                    "mean_difference": float(diff.mean()),
                    "p_value_raw": max(p, np.nextafter(0, 1)),
                    "effect_size": dz,
                })
    if rows:
        from src.experiments.final_publication_scenarios.statistics import holm_adjust
        raw = [r["p_value_raw"] for r in rows]
        adj = holm_adjust(raw)
        for i, r in enumerate(rows):
            r["p_value_holm"] = adj[i]
    return pd.DataFrame(rows)


def _aggregate_graph_stats(campaign_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted((campaign_root / "runs").iterdir()):
        gs = run_dir / "graph_statistics.csv"
        if gs.exists():
            r = pd.read_csv(gs).iloc[0].to_dict()
            r["run_id"] = run_dir.name
            rows.append(r)
    return pd.DataFrame(rows)
