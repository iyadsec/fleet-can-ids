"""Paired statistical tests with Holm correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _paired_test(differences: np.ndarray) -> tuple[str, float, float]:
    differences = differences[np.isfinite(differences)]
    if len(differences) < 3:
        return "insufficient_data", float("nan"), float("nan")
    if np.allclose(differences, 0):
        return "wilcoxon", 1.0, 0.0
    try:
        _, norm_p = stats.shapiro(differences)
    except Exception:
        norm_p = 0.0
    if norm_p >= 0.05:
        stat = stats.ttest_rel(differences + 1e-12, np.zeros_like(differences))
        effect = float(np.mean(differences) / (np.std(differences, ddof=1) + 1e-9))
        return "paired_t_test", float(stat.pvalue), effect
    stat = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    effect = float(np.median(differences))
    return "wilcoxon", float(stat.pvalue), effect


def holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [1.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank))
    return adjusted


def run_paired_comparisons(
    run_level_metrics: pd.DataFrame,
    *,
    reference_method: str = "fcgnn",
    compare_methods: list[str] | None = None,
    metrics: list[str] | None = None,
    scenario_key: str | None = None,
) -> pd.DataFrame:
    compare_methods = compare_methods or ["local_ids", "descriptor_clustering", "standard_gnn"]
    metrics = metrics or ["recall", "f1", "fpr", "campaign_f1", "campaign_detection_rate"]
    df = run_level_metrics.copy()
    if scenario_key:
        df = df[df["scenario_key"] == scenario_key]

    rows: list[dict] = []
    for metric in metrics:
        for other in compare_methods:
            if other == reference_method:
                continue
            pivot = df.pivot_table(
                index=["seed", "campaign_size", "coordination_strength"],
                columns="method",
                values=metric,
                aggfunc="first",
            )
            if reference_method not in pivot.columns or other not in pivot.columns:
                continue
            diff = (pivot[reference_method] - pivot[other]).dropna().to_numpy()
            test_name, p_value, effect = _paired_test(diff)
            ci_low, ci_high = np.nan, np.nan
            if len(diff) >= 2:
                ci_low, ci_high = stats.t.interval(
                    0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                )
            rows.append(
                {
                    "scenario_key": scenario_key or "all",
                    "metric": metric,
                    "reference_method": reference_method,
                    "compare_method": other,
                    "n_pairs": int(len(diff)),
                    "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                    "test": test_name,
                    "p_value": p_value,
                    "effect_size": effect,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_value_holm"] = holm_correction(out["p_value"].fillna(1.0).tolist())
    return out
