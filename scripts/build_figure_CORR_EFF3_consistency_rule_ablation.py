#!/usr/bin/env python3
"""Regenerate figure_CORR_EFF3 — single-series fleet scenario outcomes (CTT)."""

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


def load_final_row() -> dict:
    path = next((p for p in DATA_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("CORR_EFF3_consistency_rule_ablation.csv not found")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    # Final fleet-layer outcomes (with consistency rule applied in pipeline)
    return next(
        r for r in rows
        if "with" in r["rule_state"].lower() and "without" not in r["rule_state"].lower()
    )


def build_figure(row: dict) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    metrics = [
        ("Unrelated\nincorrect merge", "unrelated_incorrect_merge_rate"),
        ("Strong\ncampaign F1", "strong_campaign_f1"),
        ("Weak\ncampaign F1", "weak_campaign_f1"),
        ("Benign false\ncampaign", "benign_false_campaign"),
    ]

    values = [float(row[key]) for _, key in metrics]
    labels = [m[0] for m in metrics]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(metrics))
    bars = ax.bar(x, values, width=0.55, color="#2c6eab", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("Rate / F1 score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Fleet correlation outcomes")
    ax.grid(axis="y", alpha=0.35, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure) -> None:
    for out_dir in OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "pdf" if out_dir.name == "figures_pdf" else "png"
        path = out_dir / f"{STEM}.{ext}"
        fig.savefig(path, format=ext, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")


def main() -> int:
    fig = build_figure(load_final_row())
    save_figure(fig)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
