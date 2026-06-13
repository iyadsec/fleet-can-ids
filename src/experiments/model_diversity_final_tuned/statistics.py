"""Paired seed-level statistical tests with Holm correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan"), "insufficient_n", float("nan")
    diff = a - b
    if np.allclose(diff, diff[0]):
        return 0.0, "paired_t", 0.0
    shapiro = stats.shapiro(diff)
    normal = shapiro.pvalue > 0.05 if len(diff) <= 5000 else True
    if normal:
        t, p = stats.ttest_rel(a, b)
        dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) else 0.0
        return float(p), "paired_t", dz
    w, p = stats.wilcoxon(a, b)
    n = len(diff)
    rbc = 1 - (2 * w) / (n * (n + 1) / 2)
    return float(p), "wilcoxon", float(rbc)


def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (m - rank))
    return adjusted


def _ci_mean(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    h = se * stats.t.ppf(1 - alpha / 2, len(values) - 1)
    return mean - h, mean + h


def run_statistical_families(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prod = metrics[metrics.get("is_dry_test", False) != True].copy() if "is_dry_test" in metrics.columns else metrics.copy()
    prod = prod[~prod["run_id"].astype(str).str.startswith("dry_")] if "run_id" in prod.columns else prod

    metric_cols = [
        "campaign_precision", "campaign_recall", "campaign_f1",
        "campaign_membership_precision", "campaign_membership_recall",
        "benign_vehicles_included", "false_campaign_alert_rate",
    ]

    families = [
        ("A_controlled_strong_HK", ("strong", 2), ("descriptor_clustering", "fcgnn")),
        ("B_weak_D1_D2", ("weak", 2), ("descriptor_clustering", "fcgnn")),
    ]

    raw_p: list[float] = []
    for family, (strength, div), (m2, m3) in families:
        sub = prod[(prod["attack_strength"] == strength) & (prod["diversity_level"] == div)]
        for col in metric_cols:
            if col not in sub.columns:
                continue
            c2 = sub[sub["method"] == m2].groupby("seed")[col].mean()
            c3 = sub[sub["method"] == m3].groupby("seed")[col].mean()
            joined = pd.concat([c2, c3], axis=1, keys=["c2", "c3"]).dropna()
            if joined.empty:
                continue
            p, test, effect = _paired_test(joined["c2"].to_numpy(), joined["c3"].to_numpy())
            lo, hi = _ci_mean((joined["c3"] - joined["c2"]).to_numpy())
            raw_p.append(p if np.isfinite(p) else 1.0)
            rows.append({
                "family": family,
                "comparison": "C3_vs_C2",
                "metric": col,
                "test": test,
                "p_value_raw": p if p > 0 else np.nextafter(0, 1),
                "effect_size": effect,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_seeds": len(joined),
            })

    if raw_p:
        adj = _holm_adjust(raw_p)
        for i, row in enumerate(rows):
            row["p_value_holm"] = adj[i] if i < len(adj) else row["p_value_raw"]

    return pd.DataFrame(rows)
