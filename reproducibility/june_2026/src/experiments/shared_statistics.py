"""Shared paired statistical tests with zero-variance handling."""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (m - rank))
    return adjusted


def paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, str, float, bool]:
    """Return (p_value, test_name, effect_size, significant).

    Identical paired samples → p=1.0, effect=0, significant=False.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan"), "insufficient_n", float("nan"), False
    diff = a - b
    if np.allclose(diff, 0, atol=1e-12, rtol=0):
        return 1.0, "not_applicable", 0.0, False
    if np.std(diff, ddof=1) == 0:
        return 1.0, "not_applicable", 0.0, False
    _, p = sp_stats.ttest_rel(a, b)
    p = float(max(p, np.nextafter(0, 1)))
    dz = float(diff.mean() / diff.std(ddof=1))
    return p, "paired_t", dz, p < 0.05


def wilcoxon_paired_test(a: np.ndarray, b: np.ndarray) -> tuple[float, str, float, bool]:
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return float("nan"), "insufficient_n", float("nan"), False
    diff = a - b
    if np.allclose(diff, 0, atol=1e-12, rtol=0):
        return 1.0, "not_applicable", 0.0, False
    try:
        stat, p = sp_stats.wilcoxon(a, b, zero_method="ztest")
    except ValueError:
        return 1.0, "not_applicable", 0.0, False
    p = float(max(p, np.nextafter(0, 1)))
    n = len(diff)
    r = 1.0 - (2.0 * stat) / (n * (n + 1)) if n else 0.0
    return p, "wilcoxon", float(r), p < 0.05
