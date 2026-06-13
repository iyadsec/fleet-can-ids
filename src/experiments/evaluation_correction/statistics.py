"""Corrected statistical testing with hypothesis families and p-value formatting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.statistical_testing import _paired_test, holm_correction

FAMILY_A: list[dict[str, Any]] = [
    {"comparison": "M4 vs M2", "ref": "fcgnn", "other": "descriptor_clustering", "metrics": ["campaign_detection_rate", "campaign_f1"]},
    {"comparison": "M4 vs M3", "ref": "fcgnn", "other": "standard_gnn", "metrics": ["campaign_detection_rate", "campaign_f1"]},
]
FAMILY_A_METRICS_EXTRA = ["campaign_precision"]

FAMILY_B: list[dict[str, Any]] = [
    {"comparison": "M4 vs M2", "ref": "fcgnn", "other": "descriptor_clustering"},
    {"comparison": "M4 vs M3", "ref": "fcgnn", "other": "standard_gnn"},
]
FAMILY_B_METRICS = [
    "recall",
    "f1",
    "fpr",
    "campaign_detection_rate",
    "campaign_f1",
    "vehicle_precision",
]

FAMILY_C_METRICS = [
    "campaign_detection_rate",
    "campaign_f1",
    "recall",
    "runtime_graph_construction_sec",
    "runtime_gnn_inference_sec",
]


def format_p_value(p: float) -> str:
    if pd.isna(p):
        return "N/A"
    if p == 0.0 or p < 1e-300:
        return "p < 0.001"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def _membership_purity_proxy(df: pd.DataFrame) -> pd.Series:
    """Per-run campaign precision proxy from vehicle false inclusions."""
    if "campaign_precision" in df.columns:
        return df["campaign_precision"]
    return pd.Series(np.nan, index=df.index)


def _run_family_tests(
    df: pd.DataFrame,
    *,
    family: str,
    comparisons: list[dict[str, Any]],
    metrics: list[str],
    group_cols: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strength in sorted(df["attack_strength"].dropna().unique()):
        for cs in sorted(df["campaign_size"].dropna().unique()):
            sub = df[(df["attack_strength"] == strength) & (df["campaign_size"] == cs)]
            for comp in comparisons:
                ref = comp["ref"]
                other = comp["other"]
                comp_metrics = comp.get("metrics", metrics)
                for metric in comp_metrics:
                    if metric == "membership_purity":
                        pivot_data = sub.copy()
                        pivot_data["membership_purity"] = _membership_purity_proxy(sub)
                        metric_col = "membership_purity"
                    elif metric not in sub.columns:
                        continue
                    else:
                        pivot_data = sub
                        metric_col = metric
                    pivot = pivot_data.pivot_table(index="seed", columns="method", values=metric_col, aggfunc="first")
                    if ref not in pivot.columns or other not in pivot.columns:
                        continue
                    diff = (pivot[ref] - pivot[other]).dropna().to_numpy()
                    test_name, p_value, effect, effect_type = _paired_test(diff)
                    ci_low, ci_high = np.nan, np.nan
                    if len(diff) >= 2:
                        ci_low, ci_high = stats.t.interval(
                            0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                        )
                    rows.append(
                        {
                            "hypothesis_family": family,
                            "attack_strength": strength,
                            "campaign_size": int(cs),
                            "comparison": comp.get("comparison", f"{ref} vs {other}"),
                            "reference_method": ref,
                            "compare_method": other,
                            "metric": metric,
                            "paired_sample_count": int(len(diff)),
                            "test_used": test_name,
                            "raw_p_value": p_value,
                            "mean_paired_difference": float(np.mean(diff)) if len(diff) else np.nan,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
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
    out["raw_p_value_formatted"] = out["raw_p_value"].map(format_p_value)
    out["adjusted_p_value_formatted"] = out["adjusted_p_value"].map(format_p_value)
    return out


def run_corrected_statistical_tests(df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    fam_a_metrics = ["campaign_detection_rate", "campaign_f1", "membership_purity"]
    fam_a = _run_family_tests(
        df,
        family="A_strong_campaign",
        comparisons=[
            {**c, "metrics": fam_a_metrics} for c in FAMILY_A
        ],
        metrics=fam_a_metrics,
        group_cols=["attack_strength", "campaign_size"],
        alpha=alpha,
    )
    fam_b = _run_family_tests(
        df,
        family="B_weak_campaign",
        comparisons=[{**c, "metrics": FAMILY_B_METRICS} for c in FAMILY_B],
        metrics=FAMILY_B_METRICS,
        group_cols=["attack_strength", "campaign_size"],
        alpha=alpha,
    )
    fam_c_rows: list[dict] = []
    fcgnn = df[df["method"] == "fcgnn"]
    for strength in sorted(fcgnn["attack_strength"].dropna().unique()):
        for metric in FAMILY_C_METRICS:
            if metric not in fcgnn.columns:
                continue
            pivot = fcgnn[fcgnn["attack_strength"] == strength].pivot_table(
                index="seed", columns="campaign_size", values=metric, aggfunc="first"
            )
            for size_a, size_b in ((2, 5), (2, 10), (5, 10)):
                if size_a not in pivot.columns or size_b not in pivot.columns:
                    continue
                diff = (pivot[size_b] - pivot[size_a]).dropna().to_numpy()
                test_name, p_value, effect, effect_type = _paired_test(diff)
                ci_low, ci_high = np.nan, np.nan
                if len(diff) >= 2:
                    ci_low, ci_high = stats.t.interval(
                        0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                    )
                fam_c_rows.append(
                    {
                        "hypothesis_family": "C_fcgnn_campaign_size_trend",
                        "attack_strength": strength,
                        "campaign_size": f"{size_a}_to_{size_b}",
                        "comparison": f"FCGNN size {size_a} vs {size_b}",
                        "reference_method": "fcgnn",
                        "compare_method": f"size_{size_b}_vs_{size_a}",
                        "metric": metric,
                        "paired_sample_count": int(len(diff)),
                        "test_used": test_name,
                        "raw_p_value": p_value,
                        "mean_paired_difference": float(np.mean(diff)) if len(diff) else np.nan,
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "effect_size": effect,
                        "effect_size_type": effect_type,
                    }
                )
    fam_c = pd.DataFrame(fam_c_rows)
    if not fam_c.empty:
        fam_c["adjusted_p_value"] = holm_correction(fam_c["raw_p_value"].fillna(1.0).tolist())
        fam_c["significant"] = fam_c["adjusted_p_value"] < alpha
        fam_c["raw_p_value_formatted"] = fam_c["raw_p_value"].map(format_p_value)
        fam_c["adjusted_p_value_formatted"] = fam_c["adjusted_p_value"].map(format_p_value)
    return pd.concat([fam_a, fam_b, fam_c], ignore_index=True)
