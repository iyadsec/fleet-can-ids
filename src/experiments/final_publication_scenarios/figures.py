"""Publication figures F1–F6."""

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


def generate_all_figures(
    fig_dir: Path,
    *,
    weak: pd.DataFrame,
    campaign_fcgnn: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> list[str]:
    generated = []
    if not weak.empty and "weak_attacked_vehicles_correlated" in weak.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        weak.groupby("seed")["weak_attacked_vehicles_correlated"].mean().plot(kind="bar", ax=ax, alpha=0.7, label="weak attacked correlated")
        if "benign_weak_signals_incorrectly_included" in weak.columns:
            weak.groupby("seed")["benign_weak_signals_incorrectly_included"].mean().plot(ax=ax, alpha=0.7, label="benign weak included")
        ax.legend()
        ax.set_title("Weak campaign support (S4)")
        generated.append(_save(fig, fig_dir, "figure_F1_weak_campaign_support"))

    if not campaign_fcgnn.empty:
        agg = campaign_fcgnn.groupby(["attack_strength", "campaign_size"])["campaign_detection_rate"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        for strength, g in agg.groupby("attack_strength"):
            ax.plot(g["campaign_size"], g["campaign_detection_rate"], marker="o", label=strength)
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign detection rate")
        ax.legend()
        generated.append(_save(fig, fig_dir, "figure_F2_campaign_detection_vs_campaign_size"))

        agg2 = campaign_fcgnn.groupby(["attack_strength", "campaign_size"])["campaign_f1"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        for strength, g in agg2.groupby("attack_strength"):
            ax.plot(g["campaign_size"], g["campaign_f1"], marker="o", label=strength)
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign F1")
        ax.legend()
        generated.append(_save(fig, fig_dir, "figure_F3_campaign_F1_vs_campaign_size"))

    if not edge_df.empty and "unique_edges" in edge_df.columns:
        for scenario in edge_df["scenario"].dropna().unique():
            sub = edge_df[edge_df["scenario"] == scenario]
            fig, ax = plt.subplots(figsize=(7, 4))
            for _, g in sub.groupby("seed"):
                g = g.sort_values("unique_edges")
                ax.plot(g["unique_edges"], g["campaign_f1"], alpha=0.3, color="C0")
            mean = sub.groupby("unique_edges")["campaign_f1"].mean().reset_index().sort_values("unique_edges")
            ax.plot(mean["unique_edges"], mean["campaign_f1"], color="C1", linewidth=2, label="mean")
            ax.set_xlabel("Unique undirected edges")
            ax.set_ylabel("Campaign F1")
            ax.set_title(f"Campaign F1 vs edges ({scenario})")
            ax.legend()
            generated.append(_save(fig, fig_dir, f"figure_F4_campaign_F1_vs_unique_edges_{scenario}"))

        fig, ax = plt.subplots(figsize=(7, 4))
        edge_df.groupby("unique_edges")["false_campaign_alert_rate"].mean().plot(ax=ax, marker="o")
        ax.set_xlabel("Unique undirected edges")
        ax.set_ylabel("False campaign rate")
        generated.append(_save(fig, fig_dir, "figure_F5_false_campaign_rate_vs_unique_edges"))

        if "graph_build_time" in edge_df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            edge_df.groupby("unique_edges")["graph_build_time"].mean().plot(ax=ax, marker="o")
            ax.set_xlabel("Unique undirected edges")
            ax.set_ylabel("Graph build time (s)")
            generated.append(_save(fig, fig_dir, "figure_F6_runtime_vs_unique_edges"))

    return generated
