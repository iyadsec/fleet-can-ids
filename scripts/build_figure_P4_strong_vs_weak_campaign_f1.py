#!/usr/bin/env python3
"""Regenerate figure_P4_strong_vs_weak_campaign_F1 from canonical balanced F1 values."""

from __future__ import annotations

import csv
import statistics as stats
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
BUNDLE = REPO / f"experimental-{TODAY}" / "01_primary_ocslab_balanced"
METRICS = BUNDLE / "results" / "campaign_metrics.csv"
OUT_DIR = BUNDLE / "figures"
OUT_PDF = OUT_DIR / "figure_P4_strong_vs_weak_campaign_F1.pdf"
OUT_PNG = OUT_DIR / "figure_P4_strong_vs_weak_campaign_F1.png"

GROUPS = {
    "Strong": "Strong Coordinated Campaign",
    "Weak": "Weak Coordinated Campaign",
}
CAMPAIGN_SIZES = [2, 5, 10]


def load_series(metrics_path: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}
    for label, group in GROUPS.items():
        f1_means: list[float] = []
        f1_stds: list[float] = []
        for cs in CAMPAIGN_SIZES:
            sub = [r for r in rows if r["experiment_group"] == group and int(float(r["campaign_size"])) == cs]
            vals = [float(r["campaign_f1"]) for r in sub]
            f1_means.append(stats.mean(vals))
            f1_stds.append(stats.pstdev(vals) if len(vals) > 1 else 0.0)
        means[label] = f1_means
        stds[label] = f1_stds
    return means, stds


def build_figure(means: dict[str, list[float]], stds: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = CAMPAIGN_SIZES

    ax.errorbar(
        x,
        means["Strong"],
        yerr=stds["Strong"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#2c6eab",
        label="Strong",
    )
    ax.errorbar(
        x,
        means["Weak"],
        yerr=stds["Weak"],
        marker="s",
        linewidth=2,
        capsize=4,
        color="#e07a2f",
        label="Weak",
    )

    ax.set_xlabel("Campaign size")
    ax.set_ylabel("Campaign F1")
    ax.set_xticks(x)
    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Strong vs weak coordinated-campaign F1 (balanced OCSLab)")

    for cs, val in zip(x, means["Strong"], strict=True):
        ax.annotate(f"{val:.3f}", (cs, val), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#2c6eab")
    for cs, val in zip(x, means["Weak"], strict=True):
        ax.annotate(f"{val:.3f}", (cs, val), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#e07a2f")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, format="png", dpi=150)
    plt.close(fig)


def main() -> int:
    if not METRICS.is_file():
        raise FileNotFoundError(f"Missing canonical metrics: {METRICS}")
    means, stds = load_series(METRICS)
    build_figure(means, stds)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    print("Strong F1:", dict(zip(CAMPAIGN_SIZES, means["Strong"], strict=True)))
    print("Weak F1:", dict(zip(CAMPAIGN_SIZES, means["Weak"], strict=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
