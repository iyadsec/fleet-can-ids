"""Tables and figures for Phase 4 model diversity."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.model_diversity.compositions import CONFIG_LABELS


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


def export_tables(
    output_root: Path,
    *,
    run_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    local_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    unsupported_df: pd.DataFrame,
    audit_summary: dict,
) -> list[str]:
    tables_dir = output_root / "tables"
    created: list[str] = []

    d1_rows = []
    for strength in ("strong", "weak"):
        for dl in (1, 2, 3):
            supported = dl in audit_summary.get(f"supported_{strength}_diversity_levels", [])
            reason = ""
            if not supported:
                reason = "unsupported_by_dataset"
                if strength == "weak" and dl == 3:
                    reason = "no weak Chevrolet malicious descriptors"
            d1_rows.append(
                {
                    "Diversity level": f"D{dl}",
                    "Campaign size": 5,
                    "Attacked model composition": "seed-rotated (see manifests)",
                    "Attack strength": strength,
                    "Supported": "yes" if supported else "no",
                    "Reason if unsupported": reason,
                }
            )
    _bundle(pd.DataFrame(d1_rows), tables_dir / "table_D1_diversity_experiment_design", "Diversity experiment design", "table_D1")
    created.append("table_D1")

    for tag, strength in (("D2", "strong"), ("D3", "weak")):
        sub = fleet_df[(fleet_df["attack_strength"] == strength) & (fleet_df["framework_config"].isin(["C2", "C3"]))]
        if sub.empty:
            continue
        rows = []
        for (cfg, dl), g in sub.groupby(["framework_config", "diversity_level"]):
            row = {
                "Configuration": CONFIG_LABELS[cfg],
                "Diversity level": int(dl),
                "Campaign detection rate": _mss(g["campaign_detection_rate"]),
                "Campaign F1": _mss(g["campaign_f1"]),
                "Membership purity": _mss(g.get("membership_purity", g["campaign_precision"])),
                "False campaign rate": _mss(g["false_campaign_alert_rate"]),
            }
            if tag == "D2":
                row.update(
                    {
                        "Campaign precision": _mss(g["campaign_precision"]),
                        "Campaign recall": _mss(g["campaign_recall"]),
                        "Incorrect merging": _mss(g.get("incorrect_campaign_merging", pd.Series([0]))),
                    }
                )
            else:
                w = weak_df[(weak_df["attack_strength"] == strength) & (weak_df["framework_config"] == cfg) & (weak_df["diversity_level"] == dl)]
                row.update(
                    {
                        "Weak campaign members supported": int(w["weak_campaign_members_supported"].sum()) if not w.empty else 0,
                        "Weak attacked vehicles correlated": int(w["weak_attacked_vehicles_correlated"].sum()) if not w.empty else 0,
                        "Benign weak signals included": int(w["weak_benign_signals_included"].sum()) if not w.empty else 0,
                    }
                )
            rows.append(row)
        _bundle(pd.DataFrame(rows), tables_dir / f"table_{tag}_{strength}_model_diversity_results", f"{strength.title()} model-diversity results", f"table_{tag}")
        created.append(f"table_{tag}")

    if not sim_df.empty:
        d4 = sim_df.groupby("diversity_level").agg(
            within_model_malicious=("within_model_attack_similarity", "mean"),
            cross_model_malicious=("cross_model_attack_similarity", "mean"),
            cross_model_benign=("benign_cross_model_similarity", "mean"),
            campaign_gap=("malicious_minus_benign_cross_sim", "mean"),
            cross_model_edge_pct=("cross_model_edge_percentage", "mean"),
        ).reset_index()
        d4.columns = [
            "Diversity level", "Within-model malicious similarity", "Cross-model malicious similarity",
            "Cross-model benign similarity", "Campaign similarity gap", "Cross-model edge percentage",
        ]
        _bundle(d4, tables_dir / "table_D4_descriptor_portability", "Descriptor portability", "table_D4")
        created.append("table_D4")

    if not local_df.empty and "vehicle_model" not in local_df.columns:
        pass
    membership_path = output_root / "results" / "strong" / "runs"
    # Table D5 from run-level local metrics by model using vehicle_composition + local - simplified aggregate
    d5 = local_df.groupby(["attack_strength"]).agg(
        Precision=("precision", "mean"), Recall=("recall", "mean"), F1=("f1", "mean"),
        FPR=("fpr", "mean"), PR_AUC=("pr_auc", "mean"), Latency=("latency_sec", "mean"),
    ).reset_index()
    d5.insert(0, "Vehicle model", "All platforms (pooled)")
    d5.columns = ["Vehicle model", "Attack strength", "Precision", "Recall", "F1", "FPR", "PR-AUC", "Latency"]
    _bundle(d5, tables_dir / "table_D5_local_ids_stability", "Local IDS stability", "table_D5")
    created.append("table_D5")

    if not runtime_df.empty:
        d6 = runtime_df.groupby(["framework_config", "diversity_level"]).agg(
            Nodes=("graph_nodes", "mean"),
            Unique_edges=("graph_unique_undirected_edges", "mean"),
            Cross_model_edges=("cross_model_edges", "mean"),
            Cross_model_edge_pct=("cross_model_edge_percentage", "mean"),
            Graph_build=("runtime_graph_construction_sec", "mean"),
            Inference=("runtime_total_sec", "mean"),
        ).reset_index()
        d6["Configuration"] = d6["framework_config"].map(CONFIG_LABELS)
        _bundle(d6, tables_dir / "table_D6_computational_cost", "Computational cost", "table_D6")
        created.append("table_D6")

    if not stats_df.empty:
        h7 = stats_df[
            ["attack_strength", "diversity_level", "metric", "comparison", "paired_seeds", "test",
             "mean_difference", "ci95", "adjusted_p_value_formatted", "effect_size", "significant"]
        ].rename(columns={"adjusted_p_value_formatted": "Adjusted p-value", "effect_size": "Effect size"})
        _bundle(h7, tables_dir / "table_D7_statistical_tests", "Statistical tests", "table_D7")
        created.append("table_D7")

    if not unsupported_df.empty:
        unsupported_df.to_csv(output_root / "results" / "unsupported_configurations.csv", index=False)

    return created


def export_figures(output_root: Path, fleet_df: pd.DataFrame, weak_df: pd.DataFrame, sim_df: pd.DataFrame, runtime_df: pd.DataFrame) -> list[str]:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    fc = fleet_df[fleet_df["framework_config"].isin(["C2", "C3"])]

    def _save(fig, name):
        fig.savefig(fig_dir / f"{name}.pdf")
        fig.savefig(fig_dir / f"{name}.png")
        plt.close(fig)
        created.append(name)

    if not fc.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            for strength in fc["attack_strength"].unique():
                g = fc[(fc["framework_config"] == cfg) & (fc["attack_strength"] == strength)]
                by = g.groupby("diversity_level")["campaign_f1"].mean()
                ax.plot(by.index, by.values, marker="o", label=f"{CONFIG_LABELS[cfg]} ({strength})")
        ax.set_xlabel("Attacked vehicle models (diversity level)")
        ax.set_ylabel("Campaign F1")
        ax.legend(fontsize=8)
        _save(fig, "figure_D1_campaign_F1_vs_model_diversity")

        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = fc[fc["framework_config"] == cfg]
            by = g.groupby("diversity_level")["campaign_detection_rate"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Campaign detection rate")
        ax.legend()
        _save(fig, "figure_D2_campaign_detection_vs_model_diversity")

    if not sim_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        by = sim_df.groupby("diversity_level").agg(
            mal=("cross_model_attack_similarity", "mean"),
            ben=("benign_cross_model_similarity", "mean"),
        )
        x = np.arange(len(by))
        ax.bar(x - 0.15, by["mal"], 0.3, label="Cross-model malicious")
        ax.bar(x + 0.15, by["ben"], 0.3, label="Cross-model benign")
        ax.set_xticks(x)
        ax.set_xticklabels([f"D{i}" for i in by.index])
        ax.legend()
        _save(fig, "figure_D3_cross_model_similarity")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sim_df.groupby("diversity_level")["cross_model_edge_percentage"].mean())
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Cross-model edge %")
        _save(fig, "figure_D4_cross_model_edge_percentage")

    w = weak_df[weak_df["framework_config"].isin(["C2", "C3"])]
    if not w.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        agg = w.groupby(["framework_config", "diversity_level"])["weak_campaign_members_supported"].mean().unstack(0)
        agg.plot(kind="bar", ax=ax)
        ax.set_title("Weak campaign support vs diversity")
        _save(fig, "figure_D5_weak_campaign_support_vs_diversity")

    if not runtime_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in runtime_df["framework_config"].dropna().unique():
            g = runtime_df[runtime_df["framework_config"] == cfg]
            by = g.groupby("diversity_level")["runtime_total_sec"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS.get(cfg, cfg))
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("End-to-end latency (s)")
        ax.legend()
        _save(fig, "figure_D6_latency_vs_model_diversity")

    return created
