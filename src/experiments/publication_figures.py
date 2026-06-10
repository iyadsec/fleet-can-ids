"""Publication figures from validated run manifest only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.experiments.publication_metrics import aggregate_validated_metrics
from src.experiments.scenario_registry import METHOD_LABELS

METHOD_ORDER = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]
COLORS = {"local_ids": "#4472C4", "descriptor_clustering": "#ED7D31", "standard_gnn": "#70AD47", "fcgnn": "#C00000"}


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def figure_01_s4_method_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    s4 = metrics[metrics["scenario_key"] == "S4_weak_campaign"]
    if s4.empty:
        return
    g = s4.groupby("method")[["recall", "f1", "vehicle_recall", "campaign_detection_rate"]].mean().reindex(METHOD_ORDER)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, col in zip(axes, g.columns):
        vals = g[col].fillna(0)
        ax.bar(range(len(vals)), vals.values, color=[COLORS.get(m, "gray") for m in vals.index])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m)[:8] for m in vals.index], rotation=25, ha="right", fontsize=7)
        ax.set_title(col.replace("_", " "))
        ax.set_ylim(0, 1.05)
    fig.suptitle("S4 weak campaign — method comparison")
    fig.tight_layout()
    _save(fig, out_dir / "figure_01_s4_method_comparison")


def figure_02_weak_event_recovery(metrics: pd.DataFrame, out_dir: Path) -> None:
    s4 = metrics[metrics["scenario_key"] == "S4_weak_campaign"]
    if s4.empty:
        return
    g = s4.groupby("method")[["weak_malicious_recovered", "weak_benign_promoted"]].mean().reindex(METHOD_ORDER)
    x = range(len(g))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - w / 2 for i in x], g["weak_malicious_recovered"], width=w, label="Weak malicious recovered", color="#4472C4")
    ax.bar([i + w / 2 for i in x], g["weak_benign_promoted"], width=w, label="Weak benign promoted", color="#C00000")
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in g.index], rotation=15, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_title("S4 weak-event recovery trade-off")
    _save(fig, out_dir / "figure_02_weak_event_recovery")


def figure_03_coordination_strength(metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["scenario_key"].isin(["S3_strong_campaign", "S4_weak_campaign"])]
    sub = sub[sub["method"].isin(["descriptor_clustering", "standard_gnn", "fcgnn"])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for meth in ["descriptor_clustering", "standard_gnn", "fcgnn"]:
        msub = sub[sub["method"] == meth]
        if msub.empty:
            continue
        g = msub.groupby("coordination_strength")["campaign_f1"].mean()
        ax.plot(g.index, g.values, marker="o", label=METHOD_LABELS.get(meth, meth))
    ax.set_xlabel("Coordination strength")
    ax.set_ylabel("Campaign F1")
    ax.legend(fontsize=8)
    ax.set_title("Campaign F1 vs coordination strength")
    _save(fig, out_dir / "figure_03_coordination_strength")


def figure_04_edge_connectivity(edge_df: pd.DataFrame, out_dir: Path) -> None:
    if edge_df.empty or "unique_undirected_edges" not in edge_df.columns:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, g in edge_df.groupby("method"):
        ax.plot(g["unique_undirected_edges"], g["campaign_f1"], marker="o", label=METHOD_LABELS.get(meth, meth))
    ax.set_xlabel("Unique undirected edges")
    ax.set_ylabel("Campaign F1")
    ax.legend(fontsize=8)
    ax.set_title("Campaign F1 vs graph connectivity")
    _save(fig, out_dir / "figure_04a_campaign_f1_vs_edges")
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, g in edge_df.groupby("method"):
        ax.plot(g["unique_undirected_edges"], g["false_campaign_alert_rate"], marker="s", label=METHOD_LABELS.get(meth, meth))
    ax.set_xlabel("Unique undirected edges")
    ax.set_ylabel("False campaign alert rate")
    ax.legend(fontsize=8)
    _save(fig, out_dir / "figure_04b_false_campaign_rate_vs_edges")
    fig, ax = plt.subplots(figsize=(6, 4))
    for meth, g in edge_df.groupby("method"):
        ax.plot(g["unique_undirected_edges"], g["campaign_f1"], marker="o", label=f"{METHOD_LABELS.get(meth,meth)} F1")
        ax.plot(g["unique_undirected_edges"], g["false_campaign_alert_rate"], marker="x", linestyle="--", label=f"{METHOD_LABELS.get(meth,meth)} false rate")
    ax.set_xlabel("Unique undirected edges")
    ax.legend(fontsize=7)
    _save(fig, out_dir / "figure_04_edge_connectivity_tradeoff")


def figure_05_campaign_size(metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["scenario_key"].isin(["S3_strong_campaign", "S4_weak_campaign"])]
    sub = sub[sub["method"].isin(["descriptor_clustering", "standard_gnn", "fcgnn"])]
    if sub.empty or sub["campaign_size"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for sc, sty in [("S3_strong_campaign", "-"), ("S4_weak_campaign", "--")]:
        for meth in ["descriptor_clustering", "standard_gnn", "fcgnn"]:
            msub = sub[(sub["scenario_key"] == sc) & (sub["method"] == meth)]
            if msub.empty:
                continue
            g = msub.groupby("campaign_size")["campaign_detection_rate"].mean()
            ax.plot(g.index, g.values, sty + "o", label=f"{sc[:2]} {METHOD_LABELS.get(meth,meth)[:6]}")
    ax.set_xlabel("Campaign size (attacked vehicles)")
    ax.set_ylabel("Campaign detection rate")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out_dir / "figure_05_campaign_detection_vs_size")


def generate_all_figures(
    validated: pd.DataFrame,
    out_dir: Path,
    edge_df: pd.DataFrame | None = None,
) -> None:
    metrics = aggregate_validated_metrics(validated)
    figure_01_s4_method_comparison(metrics, out_dir)
    figure_02_weak_event_recovery(metrics, out_dir)
    figure_03_coordination_strength(metrics, out_dir)
    figure_04_edge_connectivity(edge_df if edge_df is not None else pd.DataFrame(), out_dir)
    figure_05_campaign_size(metrics, out_dir)
