"""Statistical tests: C3 vs C2 campaign level only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.evaluation_correction.statistics import format_p_value
from src.experiments.statistical_testing import _paired_test, holm_correction


def run_hierarchical_statistics(fleet_df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    sub = fleet_df[fleet_df["framework_config"].isin(["C2", "C3"])].copy()
    if sub.empty:
        return pd.DataFrame()

    metrics = [
        "campaign_f1",
        "campaign_detection_rate",
        "membership_purity",
        "false_campaign_alert_rate",
        "incorrect_campaign_merging",
    ]
    rows: list[dict[str, Any]] = []
    for scenario in sorted(sub["scenario_id"].dropna().unique()):
        for cs in sorted(sub["campaign_size"].dropna().unique()):
            for coord in sorted(sub["coordination_strength"].dropna().unique()):
                block = sub[
                    (sub["scenario_id"] == scenario)
                    & (sub["campaign_size"] == cs)
                    & (sub["coordination_strength"] == coord)
                ]
                for metric in metrics:
                    if metric not in block.columns:
                        continue
                    pivot = block.pivot_table(index="seed", columns="framework_config", values=metric, aggfunc="first")
                    if "C3" not in pivot.columns or "C2" not in pivot.columns:
                        continue
                    diff = (pivot["C3"] - pivot["C2"]).dropna().to_numpy()
                    test, p, effect, etype = _paired_test(diff)
                    ci_low, ci_high = np.nan, np.nan
                    if len(diff) >= 2:
                        ci_low, ci_high = stats.t.interval(
                            0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                        )
                    rows.append(
                        {
                            "scenario": scenario,
                            "campaign_size": int(cs),
                            "coordination_strength": float(coord),
                            "metric": metric,
                            "comparison": "C3 vs C2",
                            "paired_seeds": int(len(diff)),
                            "test": test,
                            "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "raw_p_value": p,
                            "effect_size": effect,
                            "effect_size_type": etype,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["adjusted_p_value"] = holm_correction(out["raw_p_value"].fillna(1.0).tolist())
    out["significant"] = out["adjusted_p_value"] < alpha
    out["adjusted_p_value_formatted"] = out["adjusted_p_value"].map(format_p_value)
    out["raw_p_value_formatted"] = out["raw_p_value"].map(format_p_value)
    return out
