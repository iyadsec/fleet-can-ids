"""Tables and figures for hierarchical alignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.experiments.hierarchical_alignment.transform import CONFIG_LABELS


def _mss(s: pd.Series, d: int = 3) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "N/A"
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def _bundle(df: pd.DataFrame, stem: Path, caption: str, label: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    (stem.with_suffix(".md")).write_text(f"# {caption}\n\n{df.to_markdown(index=False)}\n", encoding="utf-8")
    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            df.to_latex(index=False, escape=True, float_format="%.4f"),
            "\\end{table}",
        ]
    )
    (stem.with_suffix(".tex")).write_text(tex, encoding="utf-8")


def table_h1_definitions() -> pd.DataFrame:
    rows = [
        ("Vehicle local", "local_anomaly_score", "Isolation Forest", "Per-window anomaly score", "Event ROC/PR"),
        ("Vehicle local", "local_evidence_level", "Isolation Forest", "benign / weak / strong band", "Interpretation"),
        ("Vehicle local", "local_event_alert", "Isolation Forest", "Strong local alert (not fleet)", "Event precision/recall"),
        ("Fleet", "fleet_cluster_id", "DBSCAN on similarity or embedding", "Behavioural cluster id", "Campaign structure"),
        ("Fleet", "fleet_campaign_member", "Cluster qualification gate", "In coordinated campaign cluster", "Campaign detection"),
        ("Fleet", "fleet_campaign_confidence", "GNN score + cohesion", "Campaign-support confidence", "Analysis only"),
        ("Fleet", "fleet_decision", "Fleet correlation layer", "isolated / coordinated / none", "Campaign evaluation"),
    ]
    return pd.DataFrame(rows, columns=["Layer", "Output", "Produced by", "Meaning", "Used for evaluation"])


def export_tables(
    local_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    capability_df: pd.DataFrame,
    tables_dir: Path,
    *,
    events_df: pd.DataFrame | None = None,
) -> list[str]:
    created = []
    _bundle(table_h1_definitions(), tables_dir / "table_H1_hierarchical_output_definitions", "Hierarchical output definitions", "table_H1")
    created.append("table_H1")

    h2 = local_df.groupby("scenario_id").agg(
        Precision=("precision", "mean"),
        Recall=("recall", "mean"),
        F1=("f1", "mean"),
        FPR=("fpr", "mean"),
        PR_AUC=("pr_auc", "mean"),
        Latency=("latency_sec", "mean"),
    ).reset_index()
    h2.columns = ["Scenario", "Precision", "Recall", "F1", "FPR", "PR-AUC", "Latency"]
    _bundle(h2, tables_dir / "table_H2_local_ids_performance", "Local IDS performance (Isolation Forest only)", "table_H2")
    created.append("table_H2")

    s02 = fleet_df[fleet_df["scenario_id"].isin(["S0", "S1", "S2"])]
    if not s02.empty:
        expected_map = {
            "S0": "no fleet evidence",
            "S1": "isolated incident",
            "S2": "unrelated multi-vehicle incidents",
        }

        def _fleet_result(scenario: str, cfg: str) -> str:
            g = s02[(s02["scenario_id"] == scenario) & (s02["framework_config"] == cfg)]
            if g.empty:
                return "N/A"
            fcr = float(g["false_campaign_alert_rate"].mean())
            if scenario == "S0":
                return "no fleet evidence" if fcr < 0.5 else "false campaign alert"
            if scenario == "S1":
                return "isolated incident" if fcr < 0.5 else "coordinated campaign (incorrect)"
            merge = float(g.get("incorrect_campaign_merging", pd.Series([0])).mean())
            if merge > 0.5:
                return "incorrect merging"
            return "unrelated multi-vehicle incidents"

        h3_rows = []
        for sc in ["S0", "S1", "S2"]:
            c2g = s02[(s02["scenario_id"] == sc) & (s02["framework_config"] == "C2")]
            c3g = s02[(s02["scenario_id"] == sc) & (s02["framework_config"] == "C3")]
            h3_rows.append(
                {
                    "Scenario": sc,
                    "Expected fleet decision": expected_map[sc],
                    "Similarity-only result": _fleet_result(sc, "C2"),
                    "GraphSAGE result": _fleet_result(sc, "C3"),
                    "False campaign rate": _mss(pd.concat([c2g["false_campaign_alert_rate"], c3g["false_campaign_alert_rate"]])),
                    "Incorrect merging": _mss(pd.concat([c2g.get("incorrect_campaign_merging", pd.Series([0])), c3g.get("incorrect_campaign_merging", pd.Series([0]))])),
                    "Correct decision rate": _mss(1 - pd.concat([c2g["false_campaign_alert_rate"], c3g["false_campaign_alert_rate"]])),
                }
            )
        _bundle(pd.DataFrame(h3_rows), tables_dir / "table_H3_safety_scenario_decisions", "Safety and scenario decisions (S0–S2)", "table_H3")
        created.append("table_H3")

    for scen, tag, name in (("S3", "H4", "strong"), ("S4", "H5", "weak")):
        sub = fleet_df[(fleet_df["scenario_id"] == scen) & (fleet_df["framework_config"].isin(["C2", "C3"]))]
        if sub.empty:
            continue
        rows = []
        for (cfg, cs), g in sub.groupby(["framework_config", "campaign_size"]):
            row = {
                "Campaign size": int(cs),
                "Configuration": CONFIG_LABELS[cfg],
                "Campaign detection rate": _mss(g["campaign_detection_rate"]),
                "Campaign precision": _mss(g["campaign_precision"]),
                "Campaign recall": _mss(g["campaign_recall"]),
                "Campaign F1": _mss(g["campaign_f1"]),
                "Membership purity": _mss(g.get("membership_purity", g["campaign_precision"])),
                "Fragmentation": _mss(g.get("fragmentation", pd.Series([0]))),
                "False campaign rate": _mss(g["false_campaign_alert_rate"]),
            }
            if tag == "H5":
                w = weak_df[weak_df["run_id"].isin(g["run_id"])]
                row.update(
                    {
                        "Weak campaign members supported": int(w["weak_campaign_members_supported"].sum()) if not w.empty else 0,
                        "Weak attacked vehicles correlated": int(w["weak_attacked_vehicles_correlated"].sum()) if not w.empty else 0,
                        "Benign weak signals included": int(w["weak_benign_signals_included"].sum()) if not w.empty else 0,
                    }
                )
            rows.append(row)
        _bundle(pd.DataFrame(rows), tables_dir / f"table_{tag}_{name}_campaign_correlation", f"{name.title()} campaign correlation", f"table_{tag}")
        created.append(f"table_{tag}")

    _bundle(capability_df, tables_dir / "table_H6_end_to_end_capability_comparison", "End-to-end capability comparison", "table_H6")
    created.append("table_H6")

    if not stats_df.empty:
        h7 = stats_df[
            [
                "scenario",
                "coordination_strength",
                "metric",
                "comparison",
                "paired_seeds",
                "test",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "adjusted_p_value_formatted",
                "effect_size",
                "significant",
            ]
        ].rename(
            columns={
                "scenario": "Scenario",
                "metric": "Metric",
                "comparison": "Comparison",
                "paired_seeds": "Paired seeds",
                "test": "Test",
                "mean_difference": "Mean difference",
                "ci95_low": "CI low",
                "ci95_high": "CI high",
                "adjusted_p_value_formatted": "Adjusted p-value",
                "effect_size": "Effect size",
                "significant": "Significant",
            }
        )
        _bundle(h7, tables_dir / "table_H7_statistical_comparison", "C3 vs C2 statistical comparison", "table_H7")
        created.append("table_H7")
    return created


def export_figures(
    fleet_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # H1 flow diagram
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    boxes = [
        ("Isolation Forest\nlocal evidence", "#4472C4"),
        ("Suspicious\ndescriptors", "#70AD47"),
        ("Behavioural\nfleet graph", "#FFC000"),
        ("GraphSAGE\ncorrelation", "#ED7D31"),
        ("Campaign\ndecision", "#C00000"),
    ]
    x = 0.02
    for label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.35), 0.16, 0.3, boxstyle="round", fc=color, ec="black", alpha=0.85))
        ax.text(x + 0.08, 0.5, label, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        x += 0.19
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(figures_dir / "figure_H1_hierarchical_detection_flow.pdf")
    fig.savefig(figures_dir / "figure_H1_hierarchical_detection_flow.png")
    plt.close(fig)
    created.append("figure_H1")

    fc = fleet_df[fleet_df["framework_config"].isin(["C2", "C3"])]
    if not fc.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = fc[fc["framework_config"] == cfg]
            by = g.groupby(["scenario_id", "campaign_size"])["campaign_f1"].mean().reset_index()
            for sc in by["scenario_id"].unique():
                s = by[by["scenario_id"] == sc]
                ax.plot(s["campaign_size"], s["campaign_f1"], marker="o", label=f"{CONFIG_LABELS[cfg]} ({sc})")
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign F1")
        ax.legend(fontsize=8)
        ax.set_title("Similarity-only vs GraphSAGE campaign F1")
        fig.savefig(figures_dir / "figure_H2_similarity_vs_graph_campaign_F1.pdf")
        fig.savefig(figures_dir / "figure_H2_similarity_vs_graph_campaign_F1.png")
        plt.close(fig)
        created.append("figure_H2")

    w = weak_df[weak_df["framework_config"].isin(["C2", "C3"])]
    if not w.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        agg = w.groupby("framework_config").agg(
            supported=("weak_campaign_members_supported", "mean"),
            benign=("weak_benign_signals_included", "mean"),
        )
        x = np.arange(len(agg))
        ax.bar(x - 0.15, agg["supported"], 0.3, label="Weak attacked supported")
        ax.bar(x + 0.15, agg["benign"], 0.3, label="Benign weak included")
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(i, i) for i in agg.index], rotation=15, ha="right")
        ax.legend()
        ax.set_title("Weak campaign support (S4 runs)")
        fig.savefig(figures_dir / "figure_H3_weak_campaign_support.pdf")
        fig.savefig(figures_dir / "figure_H3_weak_campaign_support.png")
        plt.close(fig)
        created.append("figure_H3")

        fig, ax = plt.subplots(figsize=(7, 5))
        s4 = fc[fc["scenario_id"] == "S4"]
        for cfg in ("C2", "C3"):
            g = s4[s4["framework_config"] == cfg]
            by = g.groupby("campaign_size")["campaign_detection_rate"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign detection rate")
        ax.legend()
        fig.savefig(figures_dir / "figure_H4_campaign_detection_vs_size.pdf")
        fig.savefig(figures_dir / "figure_H4_campaign_detection_vs_size.png")
        plt.close(fig)
        created.append("figure_H4")

    fig, ax = plt.subplots(figsize=(8, 5))
    for sc in sorted(fleet_df["scenario_id"].dropna().unique()):
        g = fleet_df[(fleet_df["scenario_id"] == sc) & (fleet_df["framework_config"].isin(["C2", "C3"]))]
        if g.empty:
            continue
        ax.bar(f"{sc}-C2", g[g["framework_config"] == "C2"]["false_campaign_alert_rate"].mean(), color="#70AD47")
        ax.bar(f"{sc}-C3", g[g["framework_config"] == "C3"]["false_campaign_alert_rate"].mean(), color="#ED7D31")
    ax.set_ylabel("False campaign alert rate")
    ax.set_title("False campaign rate by scenario")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(figures_dir / "figure_H5_false_campaign_rate.pdf")
    fig.savefig(figures_dir / "figure_H5_false_campaign_rate.png")
    plt.close(fig)
    created.append("figure_H5")

    return created
