"""Publication tables and figures for framework ablation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.framework_ablation.config import CONFIG_LABELS, FRAMEWORK_CONFIGS

CONFIG_ORDER = ["C1", "C2", "C3"]


def _mean_std(s: pd.Series, d: int = 3) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "N/A"
    if len(x) == 1:
        return f"{x.mean():.{d}f}"
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}"


def _write_bundle(df: pd.DataFrame, stem: Path, *, caption: str, label: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    md = f"# {caption}\n\n{df.to_markdown(index=False)}\n"
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
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
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")


def table_f1_configurations() -> pd.DataFrame:
    rows = []
    for cid, cfg in FRAMEWORK_CONFIGS.items():
        rows.append(
            {
                "Configuration": cid,
                "Label": cfg["label"],
                "Local detection": "Yes" if cfg["local_detection"] else "No",
                "Behavioural similarity": "Yes" if cfg["behavioural_similarity"] else "No",
                "Graph construction": "Yes" if cfg["graph_construction"] else "No",
                "Message passing": "Yes" if cfg["message_passing"] else "No",
                "Campaign decision": "Yes" if cfg["campaign_decision"] else "N/A",
            }
        )
    return pd.DataFrame(rows)


def table_f2_safety(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["scenario_id"].isin(["S0", "S1", "S2"])]
    rows = []
    for (sc, cfg), grp in sub.groupby(["scenario_id", "framework_config"]):
        rows.append(
            {
                "Scenario": sc,
                "Configuration": CONFIG_LABELS.get(cfg, cfg),
                "Event detection result": _mean_std(grp["f1"]),
                "False campaign alert rate": _mean_std(grp.get("false_campaign_alert_rate", pd.Series([np.nan]))),
                "Incorrect merging": _mean_std(grp.get("incorrect_campaign_merging", pd.Series([0]))),
                "Correct final decision rate": _mean_std(1 - grp.get("fpr", pd.Series([0]))),
            }
        )
    return pd.DataFrame(rows)


def table_f3_strong(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["scenario_id"] == "S3"]
    rows = []
    for (cfg, cs, coord), grp in sub.groupby(["framework_config", "campaign_size", "coordination_strength"]):
        if cfg not in CONFIG_ORDER:
            continue
        rows.append(
            {
                "Configuration": CONFIG_LABELS.get(cfg, cfg),
                "Campaign size": int(cs),
                "Coordination strength": float(coord),
                "Campaign detection rate": _mean_std(grp["campaign_detection_rate"]),
                "Campaign precision": _mean_std(grp["campaign_precision"]),
                "Campaign recall": _mean_std(grp["campaign_recall"]),
                "Campaign F1": _mean_std(grp["campaign_f1"]),
                "Membership purity": _mean_std(grp.get("membership_purity", grp["campaign_precision"])),
                "False campaign rate": _mean_std(grp["false_campaign_alert_rate"]),
            }
        )
    return pd.DataFrame(rows)


def table_f4_weak(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["scenario_id"] == "S4"]
    rows = []
    for (cfg, cs, coord), grp in sub.groupby(["framework_config", "campaign_size", "coordination_strength"]):
        if cfg not in CONFIG_ORDER:
            continue
        rows.append(
            {
                "Configuration": CONFIG_LABELS.get(cfg, cfg),
                "Campaign size": int(cs),
                "Coordination strength": float(coord),
                "Event recall": _mean_std(grp["recall"]),
                "Event F1": _mean_std(grp["f1"]),
                "Event FPR": _mean_std(grp["fpr"]),
                "Attacked-vehicle precision": _mean_std(grp["vehicle_precision"]),
                "Attacked-vehicle recall": _mean_std(grp["vehicle_recall"]),
                "Campaign detection rate": _mean_std(grp["campaign_detection_rate"]),
                "Campaign F1": _mean_std(grp["campaign_f1"]),
                "Weak malicious events promoted": int(grp.get("weak_malicious_promoted", pd.Series([0])).sum()),
                "Benign events incorrectly promoted": int(grp.get("benign_incorrectly_promoted", pd.Series([0])).sum()),
            }
        )
    return pd.DataFrame(rows)


def table_f5_coordination(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["scenario_id"].isin(["S3", "S4"])) & (df["campaign_size"] == 5)]
    rows = []
    for (cfg, coord), grp in sub.groupby(["framework_config", "coordination_strength"]):
        if cfg not in ("C2", "C3"):
            continue
        rows.append(
            {
                "Configuration": CONFIG_LABELS.get(cfg, cfg),
                "Coordination strength": float(coord),
                "Measured malicious cross-vehicle similarity": _mean_std(
                    grp.get("measured_cross_vehicle_similarity", pd.Series([np.nan]))
                ),
                "Campaign detection rate": _mean_std(grp["campaign_detection_rate"]),
                "Campaign F1": _mean_std(grp["campaign_f1"]),
                "Membership purity": _mean_std(grp.get("membership_purity", grp["campaign_precision"])),
                "False campaign alert rate": _mean_std(grp["false_campaign_alert_rate"]),
            }
        )
    return pd.DataFrame(rows)


def table_f7_cost(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cfg, cs, coord), grp in df[df["framework_config"].isin(CONFIG_ORDER)].groupby(
        ["framework_config", "campaign_size", "coordination_strength"]
    ):
        rows.append(
            {
                "Configuration": CONFIG_LABELS.get(cfg, cfg),
                "Campaign size": int(cs),
                "Coordination strength": float(coord),
                "Unique edges": _mean_std(grp.get("graph_unique_undirected_edges", pd.Series([np.nan]))),
                "Graph-build time (s)": _mean_std(grp.get("runtime_graph_construction_sec", pd.Series([np.nan]))),
                "Inference latency (s)": _mean_std(
                    grp.get("runtime_gnn_inference_sec", grp.get("runtime_clustering_sec", pd.Series([0])))
                ),
                "Peak memory": "N/A",
            }
        )
    return pd.DataFrame(rows)


def export_all_tables(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    tables_dir: Path,
    supp_dir: Path,
    supp_gnn: pd.DataFrame,
) -> list[str]:
    created = []
    bundles = [
        ("table_F1_framework_configurations", table_f1_configurations(), "Framework configurations"),
        ("table_F2_safety_controls", table_f2_safety(df), "Safety controls (S0–S2)"),
        ("table_F3_strong_campaign_framework_ablation", table_f3_strong(df), "Strong campaign ablation (S3)"),
        ("table_F4_weak_campaign_framework_ablation", table_f4_weak(df), "Weak campaign ablation (S4)"),
        ("table_F5_coordination_strength_sensitivity", table_f5_coordination(df), "Coordination strength sensitivity"),
        ("table_F6_primary_statistical_tests", stats, "Primary statistical tests"),
        ("table_F7_computational_cost", table_f7_cost(df), "Computational cost"),
    ]
    for name, tbl, cap in bundles:
        if tbl is None or (isinstance(tbl, pd.DataFrame) and tbl.empty):
            continue
        _write_bundle(tbl, tables_dir / name, caption=cap, label=name)
        created.append(name)
    if not supp_gnn.empty:
        s1 = supp_gnn.groupby(["attack_strength", "campaign_size"]).agg(
            campaign_f1=("campaign_f1", "mean"),
            campaign_detection_rate=("campaign_detection_rate", "mean"),
            event_f1=("f1", "mean"),
        ).reset_index()
        s1["method"] = "Standard GNN (GCN supplementary)"
        _write_bundle(s1, supp_dir / "table_S1_standard_gnn_comparison", caption="Supplementary Standard GNN", label="table_S1")
        created.append("table_S1_standard_gnn_comparison")
    return created


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def export_figures(df: pd.DataFrame, figures_dir: Path) -> list[str]:
    created: list[str] = []
    s4 = df[df["scenario_id"] == "S4"]

    if not s4.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in CONFIG_ORDER:
            g = s4[(s4["framework_config"] == cfg) & (s4["coordination_strength"] == 1.0)]
            if g.empty:
                continue
            by_cs = g.groupby("campaign_size")["f1"].agg(["mean", "std"])
            ax.errorbar(by_cs.index, by_cs["mean"], yerr=by_cs["std"], marker="o", label=CONFIG_LABELS[cfg], capsize=4)
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Event F1 (S4 weak)")
        ax.set_title("Framework comparison — weak coordinated campaigns")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_F1_framework_comparison_S4")
        created.append("figure_F1_framework_comparison_S4")

    coord = df[(df["campaign_size"] == 5) & (df["framework_config"].isin(["C2", "C3"]))]
    if not coord.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = coord[coord["framework_config"] == cfg]
            by = g.groupby("coordination_strength")["campaign_f1"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Coordination strength")
        ax.set_ylabel("Campaign F1")
        ax.set_title("Campaign F1 vs coordination strength (n=5)")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_F2_campaign_F1_vs_coordination_strength")
        created.append("figure_F2_campaign_F1_vs_coordination_strength")

        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = coord[coord["framework_config"] == cfg]
            by = g.groupby("coordination_strength")["false_campaign_alert_rate"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Coordination strength")
        ax.set_ylabel("False campaign alert rate")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_F3_false_campaign_rate_vs_coordination_strength")
        created.append("figure_F3_false_campaign_rate_vs_coordination_strength")

    if not s4.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in CONFIG_ORDER:
            g = s4[s4["framework_config"] == cfg]
            ax.bar(CONFIG_LABELS[cfg], g["recall"].mean(), yerr=g["recall"].std(), capsize=4)
        ax.set_ylabel("Event recall (S4)")
        ax.set_title("Weak event recovery")
        plt.xticks(rotation=15, ha="right")
        _save_fig(fig, figures_dir / "figure_F4_weak_event_recovery")
        created.append("figure_F4_weak_event_recovery")

        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = s4[(s4["framework_config"] == cfg) & (s4["coordination_strength"] == 1.0)]
            by = g.groupby("campaign_size")["campaign_detection_rate"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign detection rate")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_F5_campaign_detection_vs_size")
        created.append("figure_F5_campaign_detection_vs_size")

    cost = df[df["framework_config"].isin(CONFIG_ORDER)]
    if "runtime_graph_construction_sec" in cost.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        lat = cost.groupby("framework_config").agg(
            graph=("runtime_graph_construction_sec", "mean"),
            infer=("runtime_gnn_inference_sec", "mean"),
            cluster=("runtime_clustering_sec", "mean"),
        )
        x = np.arange(len(lat))
        ax.bar(x - 0.2, lat["graph"], width=0.2, label="Graph build")
        ax.bar(x, lat["infer"].fillna(0), width=0.2, label="GNN inference")
        ax.bar(x + 0.2, lat["cluster"], width=0.2, label="Clustering")
        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in lat.index], rotation=15, ha="right")
        ax.set_ylabel("Seconds (mean)")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_F6_latency_vs_configuration")
        created.append("figure_F6_latency_vs_configuration")

    return created
