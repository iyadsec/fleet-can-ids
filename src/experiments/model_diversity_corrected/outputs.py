"""Tables and figures for corrected Phase 4 model diversity."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    pool_df: pd.DataFrame,
    orig_fleet_df: pd.DataFrame | None = None,
) -> list[str]:
    tables_dir = output_root / "tables"
    created: list[str] = []

    d1_rows = []
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        sub = pool_df[pool_df["vehicle_model"] == vm] if not pool_df.empty else pd.DataFrame()
        ben = int(sub["available_benign_descriptors"].max()) if not sub.empty else 0
        mal_str = int(sub[sub["attack_strength"] == "strong"]["available_malicious_descriptors"].max()) if not sub.empty else 0
        d1_rows.append(
            {
                "Vehicle model": vm,
                "Test benign descriptors": audit_summary.get("corrected_test_benign", {}).get(vm, {}).get("test_benign_descriptors", ben),
                "Strong malicious descriptors": mal_str,
                "Eligible benign fleet": "yes" if ben > 0 else "no",
            }
        )
    _bundle(pd.DataFrame(d1_rows), tables_dir / "table_DC1_corrected_data_availability", "Corrected data availability", "table_DC1")
    created.append("table_DC1")

    comp = {"Hyundai": 5, "Kia": 5, "Chevrolet": 5}
    _bundle(
        pd.DataFrame([{"Model": k, "Benign instances": v, "Total fleet size": 20, "Campaign size": 5} for k, v in comp.items()]),
        tables_dir / "table_DC2_benign_fleet_composition",
        "Fixed heterogeneous benign fleet composition",
        "table_DC2",
    )
    created.append("table_DC2")

    ctrl = fleet_df[
        (fleet_df.get("analysis_tier", "") == "controlled_same_attack")
        & fleet_df["framework_config"].isin(["C2", "C3"])
    ]
    if not ctrl.empty:
        rows = []
        for (cfg, dl), g in ctrl.groupby(["framework_config", "diversity_level"]):
            rows.append(
                {
                    "Configuration": CONFIG_LABELS[cfg],
                    "Diversity level": int(dl),
                    "Analysis tier": "controlled same-attack (Hyundai/Kia malfunction)",
                    "Campaign F1": _mss(g["campaign_f1"]),
                    "Campaign detection rate": _mss(g["campaign_detection_rate"]),
                    "False campaign rate": _mss(g["false_campaign_alert_rate"]),
                }
            )
        _bundle(pd.DataFrame(rows), tables_dir / "table_DC3_controlled_same_attack_diversity", "Controlled same-attack diversity", "table_DC3")
        created.append("table_DC3")

    for tag, strength in (("DC4", "strong"), ("DC5", "weak")):
        sub = fleet_df[(fleet_df["attack_strength"] == strength) & (fleet_df["framework_config"].isin(["C2", "C3"]))]
        if sub.empty:
            continue
        rows = []
        for (cfg, dl, tier), g in sub.groupby(["framework_config", "diversity_level", "analysis_tier"]):
            label = "exploratory mixed-attack" if tier == "exploratory_mixed_attack" else "controlled same-attack"
            rows.append(
                {
                    "Configuration": CONFIG_LABELS[cfg],
                    "Diversity level": int(dl),
                    "Analysis tier": label,
                    "Campaign F1": _mss(g["campaign_f1"]),
                    "Membership purity": _mss(g.get("membership_purity", g["campaign_precision"])),
                    "Benign incorrectly included": _mss(g["benign_vehicles_incorrectly_included"]) if "benign_vehicles_incorrectly_included" in g.columns else "N/A",
                }
            )
        _bundle(pd.DataFrame(rows), tables_dir / f"table_{tag}_{strength}_model_diversity_results", f"{strength.title()} model-diversity results", f"table_{tag}")
        created.append(tag)

    if not sim_df.empty:
        d6 = sim_df.groupby("diversity_level").agg(
            within_model_malicious=("within_model_attack_similarity", "mean"),
            cross_model_malicious=("cross_model_attack_similarity", "mean"),
            cross_model_benign=("benign_cross_model_similarity", "mean"),
            campaign_gap=("malicious_minus_benign_cross_sim", "mean"),
            cross_model_edge_pct=("cross_model_edge_percentage", "mean"),
        ).reset_index()
        d6.columns = [
            "Diversity level", "Within-model malicious similarity", "Cross-model malicious similarity",
            "Cross-model benign similarity", "Campaign similarity gap", "Cross-model edge percentage",
        ]
        _bundle(d6, tables_dir / "table_DC6_descriptor_portability", "Descriptor portability", "table_DC6")
        created.append("table_DC6")

    err = fleet_df[fleet_df["framework_config"].isin(["C2", "C3"])]
    if not err.empty and "false_campaign_alert_rate" in err.columns:
        d7 = err.groupby(["attack_strength", "framework_config", "diversity_level"], as_index=False).agg(
            false_campaign_rate=("false_campaign_alert_rate", "mean"),
        )
        if "benign_vehicles_incorrectly_included" in err.columns:
            bi = err.groupby(["attack_strength", "framework_config", "diversity_level"], as_index=False).agg(
                benign_included=("benign_vehicles_incorrectly_included", "mean"),
            )
            d7 = d7.merge(bi, on=["attack_strength", "framework_config", "diversity_level"], how="left")
        d7["Configuration"] = d7["framework_config"].map(CONFIG_LABELS)
        _bundle(d7, tables_dir / "table_DC7_false_campaign_and_membership_errors", "False campaign and membership errors", "table_DC7")
        created.append("table_DC7")

    if not runtime_df.empty:
        d8 = runtime_df.groupby(["framework_config", "diversity_level"]).agg(
            Nodes=("graph_nodes", "mean"),
            Unique_edges=("graph_unique_undirected_edges", "mean"),
            Graph_build=("runtime_graph_construction_sec", "mean"),
            Inference=("runtime_total_sec", "mean"),
        ).reset_index()
        d8["Configuration"] = d8["framework_config"].map(CONFIG_LABELS)
        _bundle(d8, tables_dir / "table_DC8_computational_cost", "Computational cost", "table_DC8")
        created.append("table_DC8")

    if not stats_df.empty:
        h9 = stats_df.rename(columns={"adjusted_p_value_formatted": "Adjusted p-value", "effect_size": "Effect size"})
        _bundle(h9, tables_dir / "table_DC9_statistical_tests", "Statistical tests", "table_DC9")
        created.append("table_DC9")

    if orig_fleet_df is not None and not orig_fleet_df.empty and not fleet_df.empty:
        rows = []
        for dl in sorted(fleet_df["diversity_level"].dropna().unique()):
            for strength in ("strong", "weak"):
                for metric in ("campaign_f1", "false_campaign_alert_rate", "benign_vehicles_incorrectly_included"):
                    o = orig_fleet_df[
                        (orig_fleet_df["framework_config"] == "C3")
                        & (orig_fleet_df["attack_strength"] == strength)
                        & (orig_fleet_df["diversity_level"] == dl)
                    ][metric].mean() if metric in orig_fleet_df.columns else float("nan")
                    c = fleet_df[
                        (fleet_df["framework_config"] == "C3")
                        & (fleet_df["attack_strength"] == strength)
                        & (fleet_df["diversity_level"] == dl)
                    ][metric].mean() if metric in fleet_df.columns else float("nan")
                    rows.append({"Diversity level": int(dl), "Attack strength": strength, "Metric": metric, "Original C3": o, "Corrected C3": c})
        _bundle(pd.DataFrame(rows), tables_dir / "table_DC10_original_vs_corrected_phase4", "Original vs corrected Phase 4", "table_DC10")
        created.append("table_DC10")

    if not unsupported_df.empty:
        unsupported_df.to_csv(output_root / "results/unsupported_configurations.csv", index=False)

    return created


def export_figures(
    output_root: Path,
    fleet_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    orig_fleet_df: pd.DataFrame | None = None,
) -> list[str]:
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
        _save(fig, "figure_DC1_campaign_F1_vs_model_diversity")

        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in ("C2", "C3"):
            g = fc[fc["framework_config"] == cfg]
            by = g.groupby("diversity_level")["campaign_detection_rate"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS[cfg])
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Campaign detection rate")
        ax.legend()
        _save(fig, "figure_DC2_campaign_detection_vs_model_diversity")

        fig, ax = plt.subplots(figsize=(7, 5))
        g = fc[fc["framework_config"] == "C3"]
        by = g.groupby("diversity_level")["false_campaign_alert_rate"].mean()
        ax.plot(by.index, by.values, marker="s", color="crimson")
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("False campaign alert rate")
        _save(fig, "figure_DC5_false_campaign_rate_vs_diversity")

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
        _save(fig, "figure_DC3_cross_model_malicious_vs_benign_similarity")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sim_df.groupby("diversity_level")["cross_model_edge_percentage"].mean())
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Cross-model edge %")
        _save(fig, "figure_DC4_cross_model_edge_quality")

    if not runtime_df.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg in runtime_df["framework_config"].dropna().unique():
            g = runtime_df[runtime_df["framework_config"] == cfg]
            by = g.groupby("diversity_level")["runtime_total_sec"].mean()
            ax.plot(by.index, by.values, marker="o", label=CONFIG_LABELS.get(cfg, cfg))
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("End-to-end latency (s)")
        ax.legend()
        _save(fig, "figure_DC6_latency_vs_diversity")

    if orig_fleet_df is not None and not orig_fleet_df.empty and not fc.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, df, style in (("Original", orig_fleet_df, "--"), ("Corrected", fleet_df, "-")):
            g = df[(df["framework_config"] == "C3") & (df["attack_strength"] == "strong")]
            by = g.groupby("diversity_level")["campaign_f1"].mean()
            ax.plot(by.index, by.values, marker="o", linestyle=style, label=label)
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Campaign F1 (C3, strong)")
        ax.legend()
        _save(fig, "figure_DC7_original_vs_corrected_phase4")

    return created
