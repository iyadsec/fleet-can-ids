#!/usr/bin/env python3
"""Build non-collaborative vs FLEET-GUARD scenario comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_MAP = {
    "benign_fleet": ("S0", "benign_fleet_control", "Benign fleet"),
    "isolated_attack": ("S1", "isolated_attack", "Isolated single-vehicle attack"),
    "unrelated_incidents": ("S2", "independent_multi_vehicle_attacks", "Independent multi-vehicle attacks"),
    "strong_campaign": ("S3", "strong_coordinated_campaign", "Strong coordinated campaign"),
    "weak_campaign": ("S4", "weak_coordinated_campaign", "Weak coordinated campaign"),
}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare non-collaborative local IDS baseline vs FLEET-GUARD."
    )
    p.add_argument(
        "--campaign-metrics",
        default="experimental-2026-06-23/01_primary_ocslab_balanced/results/campaign_metrics.csv",
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tables-dir", default="tables")
    return p.parse_args()


def _prepare_fleet_guard(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        scenario_key = str(r.get("scenario_key", ""))
        if scenario_key not in SCENARIO_MAP:
            continue
        sid, canonical, label = SCENARIO_MAP[scenario_key]
        rows.append(
            {
                "scenario_id": sid,
                "scenario_key": canonical,
                "scenario_label": label,
                "seed": int(r["seed"]),
                "campaign_size": int(r.get("campaign_size", 0)),
                "method": "FLEET-GUARD",
                "fleet_correlation_used": "Yes",
                "descriptor_sharing_used": "Yes",
                "graph_reasoning_used": "Yes",
                "local_precision": np.nan,
                "local_recall": float(r.get("local_detection_rate", np.nan)),
                "local_f1": np.nan,
                "local_fpr": np.nan,
                "local_alert_generated": bool(float(r.get("strong_candidates", 0.0)) > 0.0),
                "campaign_detection_rate": float(r.get("campaign_detection_rate", np.nan)),
                "campaign_precision": float(r.get("campaign_precision", np.nan)),
                "campaign_recall": float(r.get("campaign_recall", np.nan)),
                "campaign_f1": float(r.get("campaign_f1", np.nan)),
                "false_campaign_rate": float(r.get("false_campaign_rate", np.nan)),
                "incorrect_merge_rate": float(r.get("incorrect_merging_rate", np.nan)),
                "membership_precision": float(r.get("membership_precision", np.nan)),
                "membership_recall": float(r.get("membership_recall", np.nan)),
                "membership_f1": float(r.get("membership_f1", np.nan)),
                "fragmentation": float(r.get("fragmentation_rate", np.nan)),
                "source": "campaign_metrics.csv",
            }
        )
    return pd.DataFrame(rows)


def _prepare_non_collaborative(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        scenario_key = str(r.get("scenario_key", ""))
        if scenario_key not in SCENARIO_MAP:
            continue
        sid, canonical, label = SCENARIO_MAP[scenario_key]

        tp_raw = float(r.get("strong_candidates", 0.0))
        benign_total = float(r.get("benign_source_windows", 0.0))
        attack_total = float(r.get("malicious_source_windows", 0.0))
        tp = min(tp_raw, attack_total) if attack_total else tp_raw
        fp = min(float(r.get("benign_incorrectly_promoted", 0.0)), benign_total) if benign_total else float(r.get("benign_incorrectly_promoted", 0.0))
        fn = max(attack_total - tp, 0.0)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, attack_total)
        f1 = _f1(precision, recall)
        fpr = _safe_div(fp, benign_total)

        rows.append(
            {
                "scenario_id": sid,
                "scenario_key": canonical,
                "scenario_label": label,
                "seed": int(r["seed"]),
                "campaign_size": int(r.get("campaign_size", 0)),
                "method": "Non-collaborative local IDS",
                "fleet_correlation_used": "No",
                "descriptor_sharing_used": "No",
                "graph_reasoning_used": "No",
                "local_precision": precision,
                "local_recall": recall,
                "local_f1": f1,
                "local_fpr": fpr,
                "local_alert_generated": bool(tp > 0.0),
                "campaign_detection_rate": np.nan,
                "campaign_precision": np.nan,
                "campaign_recall": np.nan,
                "campaign_f1": np.nan,
                "false_campaign_rate": np.nan,
                "incorrect_merge_rate": np.nan,
                "membership_precision": np.nan,
                "membership_recall": np.nan,
                "membership_f1": np.nan,
                "fragmentation": np.nan,
                "source": "campaign_metrics.csv (reused local strong-threshold scores)",
            }
        )
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "local_precision",
        "local_recall",
        "local_f1",
        "local_fpr",
        "campaign_detection_rate",
        "campaign_precision",
        "campaign_recall",
        "campaign_f1",
        "false_campaign_rate",
        "incorrect_merge_rate",
        "membership_precision",
        "membership_recall",
        "membership_f1",
        "fragmentation",
    ]
    out = (
        df.groupby(
            [
                "scenario_id",
                "scenario_key",
                "scenario_label",
                "method",
                "fleet_correlation_used",
                "descriptor_sharing_used",
                "graph_reasoning_used",
            ],
            as_index=False,
        )[num_cols]
        .mean(numeric_only=True)
    )
    out["local_alert_generated_rate"] = (
        df.groupby(["scenario_id", "method"])["local_alert_generated"].mean().values
    )
    return out.sort_values(["scenario_id", "method"]).reset_index(drop=True)


def _build_method_table(summary: pd.DataFrame) -> pd.DataFrame:
    fleet = summary[summary["method"] == "FLEET-GUARD"]
    baseline = summary[summary["method"] == "Non-collaborative local IDS"]

    strong_f1 = float(
        fleet.loc[fleet["scenario_key"] == "strong_coordinated_campaign", "campaign_f1"].mean()
    )
    weak_f1 = float(
        fleet.loc[fleet["scenario_key"] == "weak_coordinated_campaign", "campaign_f1"].mean()
    )
    benign_fcr = float(
        fleet.loc[fleet["scenario_key"] == "benign_fleet_control", "false_campaign_rate"].mean()
    )
    isolated_fcr = float(
        fleet.loc[fleet["scenario_key"] == "isolated_attack", "false_campaign_rate"].mean()
    )
    merge_rate = float(
        fleet.loc[
            fleet["scenario_key"] == "independent_multi_vehicle_attacks", "incorrect_merge_rate"
        ].mean()
    )

    table = pd.DataFrame(
        [
            {
                "Method": "Non-collaborative local IDS",
                "Fleet correlation used": "No",
                "Descriptor sharing used": "No",
                "Graph reasoning used": "No",
                "Strong campaign F1": "N/A",
                "Weak campaign F1": "N/A",
                "Benign false campaign rate": "N/A",
                "Isolated false campaign rate": "N/A",
                "Independent incident merge rate": "N/A",
            },
            {
                "Method": "FLEET-GUARD",
                "Fleet correlation used": "Yes",
                "Descriptor sharing used": "Yes",
                "Graph reasoning used": "Yes",
                "Strong campaign F1": f"{strong_f1:.3f}",
                "Weak campaign F1": f"{weak_f1:.3f}",
                "Benign false campaign rate": f"{benign_fcr:.3f}",
                "Isolated false campaign rate": f"{isolated_fcr:.3f}",
                "Independent incident merge rate": f"{merge_rate:.3f}",
            },
        ]
    )
    _ = baseline  # keeps intent explicit: baseline campaign metrics are intentionally N/A.
    return table


def _write_tex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_latex(index=False, escape=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    campaign_path = Path(args.campaign_metrics)
    if not campaign_path.exists():
        raise FileNotFoundError(f"Missing input: {campaign_path}")

    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    campaign_df = pd.read_csv(campaign_path)
    fleet_guard = _prepare_fleet_guard(campaign_df)
    non_collab = _prepare_non_collaborative(campaign_df)
    combined = pd.concat([fleet_guard, non_collab], ignore_index=True)
    summary = _summary(combined)

    detailed_csv = results_dir / "non_collaborative_vs_fleet_guard.csv"
    summary_csv = results_dir / "non_collaborative_vs_fleet_guard_summary.csv"
    combined.to_csv(detailed_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    method_table = _build_method_table(summary)
    method_tex = tables_dir / "table_non_collaborative_vs_fleet_guard.tex"
    scenario_tex = tables_dir / "table_scenario_level_comparison.tex"
    _write_tex(method_table, method_tex)

    scenario_table = summary.copy()
    for col in [
        "campaign_detection_rate",
        "campaign_precision",
        "campaign_recall",
        "campaign_f1",
        "false_campaign_rate",
        "incorrect_merge_rate",
        "membership_precision",
        "membership_recall",
        "membership_f1",
        "fragmentation",
    ]:
        scenario_table[col] = scenario_table[col].astype(object)
        scenario_table.loc[
            scenario_table["method"] == "Non-collaborative local IDS", col
        ] = "N/A"
    _write_tex(scenario_table, scenario_tex)

    strong_f1 = method_table.loc[method_table["Method"] == "FLEET-GUARD", "Strong campaign F1"].iloc[0]
    weak_f1 = method_table.loc[method_table["Method"] == "FLEET-GUARD", "Weak campaign F1"].iloc[0]

    print(f"Generated: {detailed_csv}")
    print(f"Generated: {summary_csv}")
    print(f"Generated: {method_tex}")
    print(f"Generated: {scenario_tex}")
    print(f"FLEET-GUARD strong campaign F1: {strong_f1}")
    print(f"FLEET-GUARD weak campaign F1: {weak_f1}")
    print(
        "Non-collaborative baseline confirmation: does not construct graphs, does not run GraphSAGE,"
        " does not run DBSCAN, and does not emit campaign clusters (campaign metrics = N/A)."
    )
    print(
        "Validation note: both methods are derived from the same scenario seeds and source rows in campaign_metrics.csv;"
        " baseline reuses the same local strong-threshold score outputs while disabling fleet reasoning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
