"""Publication figures FS1–FS10."""

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


def generate_figures(fig_dir: Path, df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    out = []

    if "transmission_percentage" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        g = df.groupby("scenario_id")["transmission_percentage"].mean()
        ax.bar(g.index, g.values)
        ax.set_ylabel("Transmission %")
        out.append(_save(fig, fig_dir, "figure_FS1_candidate_transmission_by_scenario"))

    for sid, fname, ylabel in [
        ("S3", "figure_FS2_campaign_precision_recall_vs_size", "Rate"),
        ("S3", "figure_FS4_membership_precision_recall_vs_size", "Rate"),
        ("S3", "figure_FS5_fragmentation_strong_vs_weak", "Fragmentation"),
    ]:
        sub = df[df["scenario_id"] == sid]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        g = sub.groupby("campaign_size")
        if "campaign" in fname:
            ax.plot(g["campaign_precision"].mean().index, g["campaign_precision"].mean(), marker="o", label="precision")
            ax.plot(g["campaign_recall"].mean().index, g["campaign_recall"].mean(), marker="s", linestyle="--", label="recall")
        elif "membership" in fname:
            ax.plot(g["membership_precision"].mean().index, g["membership_precision"].mean(), marker="o", label="prec")
            ax.plot(g["membership_recall"].mean().index, g["membership_recall"].mean(), marker="s", linestyle="--", label="rec")
        else:
            ax.plot(g["fragmentation_rate"].mean().index, g["fragmentation_rate"].mean(), marker="o")
        ax.legend(fontsize=8)
        ax.set_xlabel("Campaign size")
        out.append(_save(fig, fig_dir, fname))

    fig, ax = plt.subplots(figsize=(7, 4))
    for sid, style in [("S3", "-"), ("S4", "--")]:
        sub = df[df["scenario_id"] == sid]
        if sub.empty:
            continue
        g = sub.groupby("campaign_size")["campaign_f1"].mean()
        ax.plot(g.index, g.values, marker="o", linestyle=style, label=sid)
    ax.legend()
    ax.set_xlabel("Campaign size")
    ax.set_ylabel("Campaign F1")
    out.append(_save(fig, fig_dir, "figure_FS3_campaign_F1_strong_vs_weak"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for sid in ["S3", "S4"]:
        sub = df[df["scenario_id"] == sid]
        if sub.empty:
            continue
        g = sub.groupby("campaign_size")["fragmentation_rate"].mean()
        ax.plot(g.index, g.values, marker="o", label=sid)
    ax.legend()
    out.append(_save(fig, fig_dir, "figure_FS5_fragmentation_strong_vs_weak"))

    if "benign_candidates" in df.columns and "benign_vehicles_included" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        g = df.groupby("scenario_id")[["benign_candidates", "benign_vehicles_included"]].mean()
        x = range(len(g))
        ax.bar([i - 0.2 for i in x], g["benign_candidates"], width=0.4, label="benign candidates")
        ax.bar([i + 0.2 for i in x], g["benign_vehicles_included"], width=0.4, label="benign in campaign")
        ax.set_xticks(list(x))
        ax.set_xticklabels(g.index)
        ax.legend(fontsize=8)
        out.append(_save(fig, fig_dir, "figure_FS6_benign_candidates_and_members"))

    fig, ax = plt.subplots(figsize=(7, 4))
    g = df.groupby(["scenario_id", "campaign_size"])[["graph_nodes", "unique_edges"]].mean().reset_index()
    for col, mk in [("graph_nodes", "o"), ("unique_edges", "s")]:
        if col in g.columns:
            for sid in g["scenario_id"].unique():
                s = g[g["scenario_id"] == sid]
                ax.plot(s["campaign_size"], s[col], marker=mk, label=f"{sid} {col}")
    ax.legend(fontsize=7)
    out.append(_save(fig, fig_dir, "figure_FS7_graph_nodes_and_edges"))

    if "transmission_percentage" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        g = df.groupby("scenario_id")["transmission_percentage"].mean()
        ax.bar(g.index.astype(str), 100 - g.values)
        ax.set_ylabel("Bandwidth reduction %")
        out.append(_save(fig, fig_dir, "figure_FS8_bandwidth_reduction"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df.get("graph_nodes", df.get("candidate_nodes", 0)), df.get("end_to_end_latency", 0), alpha=0.6)
    ax.set_xlabel("Graph nodes")
    ax.set_ylabel("Latency (s)")
    out.append(_save(fig, fig_dir, "figure_FS9_runtime_vs_graph_nodes"))

    s4 = df[df["scenario_id"] == "S4"]
    if not s4.empty and "membership_recall" in s4.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(s4.groupby("campaign_size")["membership_recall"].mean().index,
                s4.groupby("campaign_size")["membership_recall"].mean(), marker="o")
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Weak campaign membership recall")
        out.append(_save(fig, fig_dir, "figure_FS10_weak_campaign_support"))

    return out
