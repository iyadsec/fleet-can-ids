"""Figures CR1–CR8."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, fig_dir: Path, name: str) -> str:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_figures(fig_dir: Path, all_df: pd.DataFrame) -> list[str]:
    if all_df.empty:
        return []
    generated = []
    designs = ["D0", "D1", "D2"]
    colors = {"D0": "C0", "D1": "C1", "D2": "C2"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in designs:
        sub = all_df[all_df["design"] == d].groupby("campaign_size").mean(numeric_only=True).reset_index()
        if sub.empty:
            continue
        ax.plot(sub["campaign_size"], sub["campaign_precision"], marker="o", label=f"{d} precision", color=colors[d])
        ax.plot(sub["campaign_size"], sub["campaign_recall"], marker="s", linestyle="--", label=f"{d} recall", color=colors[d], alpha=0.7)
    ax.legend(fontsize=7)
    ax.set_xlabel("Campaign size")
    generated.append(_save(fig, fig_dir, "figure_CR1_campaign_precision_recall_by_design"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in designs:
        sub = all_df[all_df["design"] == d].groupby("campaign_size")["campaign_f1"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=d)
    ax.legend()
    ax.set_xlabel("Campaign size")
    ax.set_ylabel("Campaign F1")
    generated.append(_save(fig, fig_dir, "figure_CR2_campaign_F1_by_design_and_size"))

    for metric, fname, ylabel in (
        (("membership_precision", "membership_recall"), "figure_CR3_membership_precision_recall_by_design", "Rate"),
        (("fragmentation_rate",), "figure_CR4_fragmentation_by_design_and_size", "Fragmentation rate"),
        (("benign_vehicles_included",), "figure_CR5_benign_vehicles_included", "Benign vehicles included"),
        (("cross_vehicle_campaign_edges",), "figure_CR6_cross_vehicle_edge_support", "Cross-vehicle edges"),
        (("unique_edges",), "figure_CR7_unique_edges_by_design_and_size", "Unique edges"),
        (("end_to_end_latency",), "figure_CR8_runtime_by_design_and_size", "Latency (s)"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        for d in designs:
            sub = all_df[all_df["design"] == d].groupby("campaign_size")
            if len(metric) == 2:
                m1, m2 = metric
                g = sub[[m1, m2]].mean().reset_index()
                ax.plot(g["campaign_size"], g[m1], marker="o", label=f"{d} prec")
                ax.plot(g["campaign_size"], g[m2], marker="s", linestyle="--", label=f"{d} rec")
            else:
                g = sub[metric[0]].mean()
                ax.plot(g.index, g.values, marker="o", label=d)
        ax.legend(fontsize=7)
        ax.set_xlabel("Campaign size")
        ax.set_ylabel(ylabel)
        generated.append(_save(fig, fig_dir, fname))

    return generated
