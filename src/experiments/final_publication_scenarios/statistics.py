"""Paired seed-level statistical tests with Holm correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (m - rank))
    return adjusted


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan"), "insufficient_n", float("nan")
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, "paired_t", 0.0
    _, p = sp_stats.ttest_rel(a, b)
    dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) else 0.0
    return float(max(p, np.nextafter(0, 1))), "paired_t", dz


def edge_sensitivity_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "unique_edges" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["connectivity_bin"] = pd.qcut(df["unique_edges"], q=3, duplicates="drop")
    rows = []
    for scenario in df["scenario"].dropna().unique():
        sub = df[df["scenario"] == scenario]
        bins = sub["connectivity_bin"].dropna().unique()
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            a = sub[sub["connectivity_bin"] == low].groupby("seed")["campaign_f1"].mean()
            b = sub[sub["connectivity_bin"] == high].groupby("seed")["campaign_f1"].mean()
            joined = pd.concat([a, b], axis=1, keys=["low", "high"]).dropna()
            if len(joined) < 3:
                continue
            p, test, eff = paired_test(joined["low"].to_numpy(), joined["high"].to_numpy())
            rows.append({
                "family": "edge_sensitivity",
                "experiment": scenario,
                "comparison": f"{low}_vs_{high}",
                "metric": "campaign_f1",
                "paired_seeds": len(joined),
                "mean_difference": float((joined["high"] - joined["low"]).mean()),
                "p_value_raw": p,
                "effect_size": eff,
                "test": test,
            })
    if rows:
        adj = holm_adjust([r["p_value_raw"] for r in rows])
        for i, r in enumerate(rows):
            r["p_value_holm"] = adj[i]
            r["significant"] = adj[i] < 0.05
    return pd.DataFrame(rows)
