"""Paired statistical tests with Holm correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _rank_biserial(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences)]
    n = len(differences)
    if n < 3:
        return float("nan")
    try:
        res = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        w = float(res.statistic)
        return float(1.0 - (2.0 * w) / (n * (n + 1)))
    except Exception:
        return float("nan")


def _paired_test(differences: np.ndarray) -> tuple[str, float, float, str]:
    differences = differences[np.isfinite(differences)]
    if len(differences) < 3:
        return "insufficient_data", float("nan"), float("nan"), "none"
    if np.allclose(differences, 0):
        return "wilcoxon", 1.0, 0.0, "rank_biserial"
    try:
        _, norm_p = stats.shapiro(differences)
    except Exception:
        norm_p = 0.0
    if norm_p >= 0.05:
        stat = stats.ttest_rel(differences + 1e-12, np.zeros_like(differences))
        effect = float(np.mean(differences) / (np.std(differences, ddof=1) + 1e-9))
        return "paired_t_test", float(stat.pvalue), effect, "cohens_dz"
    p = float(stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
    return "wilcoxon", p, _rank_biserial(differences), "rank_biserial"


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
            test_name, p_value, effect, _ = _paired_test(diff)
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


def run_campaign_size_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired M4 vs M1/M2/M3 per campaign size, plus FCGNN across-size tests."""
    if df.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for cs in sorted(df["campaign_size"].dropna().unique()):
        for strength in sorted(df["attack_strength"].dropna().unique()):
            sub = df[(df["campaign_size"] == cs) & (df["attack_strength"] == strength)]
            part = run_paired_comparisons(sub, scenario_key=None)
            if part.empty:
                continue
            part["campaign_size"] = int(cs)
            part["attack_strength"] = strength
            part["comparison_type"] = "method_vs_fcgnn"
            frames.append(part)

    fcgnn = df[df["method"] == "fcgnn"]
    for strength in sorted(fcgnn["attack_strength"].dropna().unique()):
        for metric in ("campaign_f1", "campaign_detection_rate"):
            pivot = fcgnn[fcgnn["attack_strength"] == strength].pivot_table(
                index="seed", columns="campaign_size", values=metric, aggfunc="first"
            )
            for size_a, size_b in ((2, 5), (2, 10), (5, 10)):
                if size_a not in pivot.columns or size_b not in pivot.columns:
                    continue
                diff = (pivot[size_b] - pivot[size_a]).dropna().to_numpy()
                test_name, p_value, effect, _ = _paired_test(diff)
                ci_low, ci_high = np.nan, np.nan
                if len(diff) >= 2:
                    ci_low, ci_high = stats.t.interval(
                        0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff)
                    )
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "comparison_type": "fcgnn_across_campaign_size",
                                "attack_strength": strength,
                                "metric": metric,
                                "campaign_size": f"{size_a}_to_{size_b}",
                                "reference_method": "fcgnn",
                                "compare_method": f"size_{size_a}_vs_{size_b}",
                                "n_pairs": int(len(diff)),
                                "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                                "test": test_name,
                                "p_value": p_value,
                                "effect_size": effect,
                                "ci95_low": float(ci_low),
                                "ci95_high": float(ci_high),
                            }
                        ]
                    )
                )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty and "p_value" in out.columns:
        out["p_value_holm"] = holm_correction(out["p_value"].fillna(1.0).tolist())
    return out


def run_corrected_phase3_primary_tests(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Primary paired comparisons for corrected Phase 3 (FCGNN vs M1/M2/M3)."""
    if df.empty:
        return pd.DataFrame()

    strong_metrics = ["campaign_detection_rate", "campaign_f1", "vehicle_recall"]
    weak_metrics = [
        "precision",
        "recall",
        "f1",
        "fpr",
        "vehicle_recall",
        "campaign_detection_rate",
        "campaign_f1",
    ]
    comparisons = [
        ("fcgnn", "descriptor_clustering", "FCGNN vs Descriptor clustering"),
        ("fcgnn", "standard_gnn", "FCGNN vs Standard GNN"),
        ("fcgnn", "local_ids", "FCGNN vs Local IDS"),
    ]
    rows: list[dict] = []

    for strength, metrics in (("strong", strong_metrics), ("weak", weak_metrics)):
        for cs in sorted(df["campaign_size"].dropna().unique()):
            sub = df[(df["attack_strength"] == strength) & (df["campaign_size"] == cs)]
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                for ref, other, label in comparisons:
                    if strength == "strong" and other == "local_ids" and metric.startswith("campaign"):
                        continue
                    pivot = sub.pivot_table(index="seed", columns="method", values=metric, aggfunc="first")
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
                            "attack_strength": strength,
                            "campaign_size": int(cs),
                            "metric": metric,
                            "comparison": label,
                            "reference_method": ref,
                            "compare_method": other,
                            "paired_seeds": int(len(diff)),
                            "statistical_test": test_name,
                            "mean_paired_difference": float(np.mean(diff)) if len(diff) else np.nan,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "raw_p_value": p_value,
                            "effect_size": effect,
                            "effect_size_type": effect_type,
                        }
                    )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_adjusted_p_value"] = holm_correction(out["raw_p_value"].fillna(1.0).tolist())
        out["significant"] = out["holm_adjusted_p_value"] < alpha
    return out
