"""Statistical analysis for CTT cross-dataset validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.utils import ensure_dir, fmt_pvalue


def holm_correction(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    if n == 0:
        return []
    order = np.argsort(pvalues)
    corrected = np.zeros(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(pvalues[idx] * (n - rank), 1.0)
    return corrected.tolist()


def compute_confidence_intervals(
    values: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    t_crit = stats.t.ppf((1 + confidence) / 2, df=max(n - 1, 1))
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
        "n": n,
    }


def run_statistical_analysis(
    scenario_results: pd.DataFrame,
    campaign_size_results: pd.DataFrame | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    stats_dir = ensure_dir(output_root / "statistics")
    ci_rows = []
    test_rows = []

    if not scenario_results.empty:
        for scenario in scenario_results["scenario"].unique():
            vals = scenario_results[scenario_results["scenario"] == scenario]["campaign_f1"].dropna().to_numpy()
            ci = compute_confidence_intervals(vals)
            ci_rows.append({"metric": "campaign_f1", "group": scenario, **ci})

        scenarios = scenario_results["scenario"].unique()
        if len(scenarios) >= 2:
            strong = scenario_results[scenario_results["scenario"] == "strong_campaign"]["campaign_f1"].dropna()
            benign = scenario_results[scenario_results["scenario"] == "benign_fleet_control"]["campaign_f1"].dropna()
            if len(strong) > 1 and len(benign) > 1:
                t_stat, p_val = stats.ttest_ind(strong, benign, equal_var=False)
                d = (strong.mean() - benign.mean()) / np.sqrt((strong.var() + benign.var()) / 2)
                test_rows.append(
                    {
                        "test": "welch_t",
                        "comparison": "strong_campaign vs benign_fleet_control",
                        "statistic": float(t_stat),
                        "p_value": fmt_pvalue(p_val),
                        "effect_size_cohens_d": float(d),
                    }
                )

    if campaign_size_results is not None and not campaign_size_results.empty:
        for size in campaign_size_results["campaign_size"].unique():
            vals = campaign_size_results[campaign_size_results["campaign_size"] == size]["campaign_f1"].dropna()
            ci = compute_confidence_intervals(vals.to_numpy())
            ci_rows.append({"metric": "campaign_f1", "group": f"size_{size}", **ci})

    if test_rows:
        raw_p = [float(r["p_value"]) if isinstance(r["p_value"], (int, float)) else 0.05 for r in test_rows]
        corrected = holm_correction(raw_p)
        for i, r in enumerate(test_rows):
            r["p_value_holm"] = fmt_pvalue(corrected[i])

    pd.DataFrame(ci_rows).to_csv(stats_dir / "confidence_intervals.csv", index=False)
    pd.DataFrame(test_rows).to_csv(stats_dir / "statistical_tests.csv", index=False)
