"""Statistical tests for framework ablation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.evaluation_correction.statistics import format_p_value
from src.experiments.statistical_testing import _paired_test, holm_correction


def run_framework_ablation_tests(df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def _add_family(
        family: str,
        sub: pd.DataFrame,
        ref_config: str,
        other_config: str,
        metrics: list[str],
        *,
        scenario: str | None = None,
        coordination_strength: float | None = None,
    ) -> None:
        for metric in metrics:
            if metric not in sub.columns:
                continue
            pivot = sub.pivot_table(
                index="seed", columns="framework_config", values=metric, aggfunc="first"
            )
            if ref_config not in pivot.columns or other_config not in pivot.columns:
                continue
            diff = (pivot[ref_config] - pivot[other_config]).dropna().to_numpy()
            test_name, p_value, effect, effect_type = _paired_test(diff)
            ci_low, ci_high = np.nan, np.nan
            if len(diff) >= 2:
                ci_low, ci_high = stats.t.interval(
                    0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                )
            rows.append(
                {
                    "hypothesis_family": family,
                    "scenario": scenario or "all",
                    "coordination_strength": coordination_strength,
                    "metric": metric,
                    "comparison": f"{ref_config} vs {other_config}",
                    "reference_config": ref_config,
                    "compare_config": other_config,
                    "paired_seeds": int(len(diff)),
                    "test": test_name,
                    "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "raw_p_value": p_value,
                    "effect_size": effect,
                    "effect_size_type": effect_type,
                }
            )

    s4 = df[(df["scenario_id"] == "S4") | (df["attack_strength"] == "weak")]
    # Family A: C3 vs C1 in S4
    fam_a_metrics = ["recall", "f1", "fpr", "vehicle_precision", "vehicle_recall"]
    for cs in sorted(s4["campaign_size"].dropna().unique()):
        for coord in sorted(s4["coordination_strength"].dropna().unique()):
            sub = s4[(s4["campaign_size"] == cs) & (s4["coordination_strength"] == coord)]
            _add_family(
                "A_C3_vs_C1_S4",
                sub,
                "C3",
                "C1",
                fam_a_metrics,
                scenario="S4",
                coordination_strength=float(coord),
            )

    # Family B: C3 vs C2 in S3 and S4
    fam_b_metrics = [
        "campaign_detection_rate",
        "campaign_f1",
        "membership_purity",
        "false_campaign_alert_rate",
        "incorrect_campaign_merging",
    ]
    for scen in ("S3", "S4"):
        sub_sc = df[df["scenario_id"] == scen]
        for cs in sorted(sub_sc["campaign_size"].dropna().unique()):
            for coord in sorted(sub_sc["coordination_strength"].dropna().unique()):
                sub = sub_sc[(sub_sc["campaign_size"] == cs) & (sub_sc["coordination_strength"] == coord)]
                _add_family(
                    "B_C3_vs_C2",
                    sub,
                    "C3",
                    "C2",
                    fam_b_metrics,
                    scenario=scen,
                    coordination_strength=float(coord),
                )

    # Family C: coordination strength effect on C2 and C3
    coord_sub = df[
        (df["scenario_id"].isin(["S3", "S4"]))
        & (df["campaign_size"] == 5)
        & (df["framework_config"].isin(["C2", "C3"]))
    ]
    fam_c_metrics = ["campaign_detection_rate", "campaign_f1", "membership_purity"]
    for config in ("C2", "C3"):
        for scen in ("S3", "S4"):
            sub = coord_sub[(coord_sub["framework_config"] == config) & (coord_sub["scenario_id"] == scen)]
            pivot = sub.pivot_table(
                index="seed", columns="coordination_strength", values="campaign_f1", aggfunc="first"
            )
            for c_a, c_b in ((0.5, 0.75), (0.5, 1.0), (0.75, 1.0)):
                if c_a not in pivot.columns or c_b not in pivot.columns:
                    continue
                diff = (pivot[c_b] - pivot[c_a]).dropna().to_numpy()
                test_name, p_value, effect, effect_type = _paired_test(diff)
                ci_low, ci_high = np.nan, np.nan
                if len(diff) >= 2:
                    ci_low, ci_high = stats.t.interval(
                        0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                    )
                rows.append(
                    {
                        "hypothesis_family": "C_coordination_strength",
                        "scenario": scen,
                        "coordination_strength": f"{c_a}_to_{c_b}",
                        "metric": "campaign_f1",
                        "comparison": f"{config} coord {c_a} vs {c_b}",
                        "reference_config": config,
                        "compare_config": f"coord_{c_b}_vs_{c_a}",
                        "paired_seeds": int(len(diff)),
                        "test": test_name,
                        "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "raw_p_value": p_value,
                        "effect_size": effect,
                        "effect_size_type": effect_type,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for fam in out["hypothesis_family"].unique():
        mask = out["hypothesis_family"] == fam
        out.loc[mask, "adjusted_p_value"] = holm_correction(
            out.loc[mask, "raw_p_value"].fillna(1.0).tolist()
        )
    out["significant"] = out["adjusted_p_value"] < alpha
    out["adjusted_p_value_formatted"] = out["adjusted_p_value"].map(format_p_value)
    out["raw_p_value_formatted"] = out["raw_p_value"].map(format_p_value)
    return out
