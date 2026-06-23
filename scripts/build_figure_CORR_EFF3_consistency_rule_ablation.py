#!/usr/bin/env python3
"""Regenerate a clearer figure_CORR_EFF3_consistency_rule_ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

DATA_CANDIDATES = [
    REPO / "OVERLEAF_CROSS_DATASET_ARTIFACTS/tables_csv/CORR_EFF3_consistency_rule_ablation.csv",
    REPO / "experimental-2026-06-23/03_cross_dataset_ctt/tables_csv/CORR_EFF3_consistency_rule_ablation.csv",
]

OUTPUT_DIRS = [
    REPO / "OVERLEAF_CROSS_DATASET_ARTIFACTS/figures_pdf",
    REPO / "OVERLEAF_CROSS_DATASET_ARTIFACTS/figures_png",
    REPO / "experimental-2026-06-23/03_cross_dataset_ctt/figures_pdf",
    REPO / "experimental-2026-06-23/03_cross_dataset_ctt/figures_png",
]

STEM = "figure_CORR_EFF3_consistency_rule_ablation"


def load_data() -> tuple[dict, dict]:
    path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("CORR_EFF3_consistency_rule_ablation.csv not found")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    without = next(r for r in rows if "without" in r["rule_state"].lower())
    with_rule = next(r for r in rows if "with" in r["rule_state"].lower() and "without" not in r["rule_state"].lower())
    return without, with_rule


def _val(row: dict, key: str) -> float:
    return float(row[key])


def build_figure(without: dict, with_rule: dict) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    metrics = [
        ("Unrelated\nincorrect merge", "unrelated_incorrect_merge_rate"),
        ("Strong\ncampaign F1", "strong_campaign_f1"),
        ("Weak\ncampaign F1", "weak_campaign_f1"),
        ("Benign false\ncampaign", "benign_false_campaign"),
    ]

    before = [_val(without, key) for _, key in metrics]
    after = [_val(with_rule, key) for _, key in metrics]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(metrics))
    width = 0.34

    color_before = "#c44e52"
    color_after = "#2ca02c"

    bars_before = ax.bar(x - width / 2, before, width, label="Without consistency rule", color=color_before, edgecolor="white", linewidth=0.8)
    bars_after = ax.bar(x + width / 2, after, width, label="With consistency rule", color=color_after, edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], ha="center")
    ax.set_ylabel("Rate / F1 score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Campaign consistency rule ablation (CTT corrected fleet layer)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.35, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def annotate_bars(bars) -> None:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    annotate_bars(bars_before)
    annotate_bars(bars_after)

    # Highlight key unrelated-merge improvement
    ax.annotate(
        "",
        xy=(x[0] + width / 2, after[0]),
        xytext=(x[0] - width / 2, before[0]),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
    )
    ax.text(
        x[0],
        1.05,
        "1.00 → 0.00",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        kind = out_dir.name
        ext = "pdf" if kind == "figures_pdf" else "png"
        path = out_dir / f"{STEM}.{ext}"
        fig.savefig(path, format=ext, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")


def main() -> int:
    without, with_rule = load_data()
    fig = build_figure(without, with_rule)
    save_figure(fig)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
