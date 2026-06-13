"""Publication tables and figures for tuned Phase 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _write_table_bundle(df: pd.DataFrame, stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    md = [f"# {title}", "", df.to_markdown(index=False)]
    stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    try:
        latex = df.to_latex(index=False, float_format="%.3f")
        stem.with_suffix(".tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass


def generate_mdf_tables(out: Path, metrics: pd.DataFrame, gate: dict, stats: pd.DataFrame, provisional: pd.DataFrame) -> list[str]:
    tables_dir = out / "tables"
    generated = []
    fleet = metrics[metrics["method"].isin(["descriptor_clustering", "fcgnn"])].copy()
    fleet["Configuration"] = fleet["framework_config"]
    fleet["Diversity level"] = fleet["diversity_level"].apply(lambda d: f"D{d}")

    for name, df, title in [
        ("table_MDF3_final_gate_configuration", pd.DataFrame([gate]), "Final gate configuration"),
        ("table_MDF5_controlled_same_attack", fleet[fleet["analysis_tier"] == "controlled_same_attack"], "Controlled same-attack campaigns"),
        ("table_MDF6_strong_diversity", fleet[(fleet["attack_strength"] == "strong") & (fleet["diversity_level"] == 3)], "Strong diversity"),
        ("table_MDF7_weak_diversity", fleet[(fleet["attack_strength"] == "weak") & (fleet["diversity_level"] == 3)], "Weak diversity"),
        ("table_MDF8_campaign_membership_quality", fleet, "Campaign membership quality"),
        ("table_MDF11_statistical_tests", stats, "Statistical tests"),
    ]:
        cols = [c for c in [
            "Configuration", "Diversity level", "attack_strength", "Campaign precision", "Campaign recall",
            "Campaign F1", "Membership precision", "Membership recall", "False campaign rate",
            "Benign vehicles included", "Fragmentation", "Incorrect merging",
            "campaign_precision", "campaign_recall", "campaign_f1",
            "campaign_membership_precision", "campaign_membership_recall",
            "false_campaign_alert_rate", "benign_vehicles_included", "fragmentation", "incorrect_merging",
        ] if c in df.columns or c.replace(" ", "_").lower() in df.columns]
        sub = df.copy()
        rename = {
            "campaign_precision": "Campaign precision",
            "campaign_recall": "Campaign recall",
            "campaign_f1": "Campaign F1",
            "campaign_membership_precision": "Membership precision",
            "campaign_membership_recall": "Membership recall",
            "false_campaign_alert_rate": "False campaign rate",
            "benign_vehicles_included": "Benign vehicles included",
            "fragmentation": "Fragmentation",
            "incorrect_merging": "Incorrect merging",
        }
        sub = sub.rename(columns=rename)
        _write_table_bundle(sub, tables_dir / name, title)
        generated.append(name)

    if not provisional.empty:
        _write_table_bundle(provisional, tables_dir / "table_MDF12_provisional_vs_tuned", "Provisional vs tuned")
        generated.append("table_MDF12_provisional_vs_tuned")

    return generated


def generate_mdf_figures(out: Path, metrics: pd.DataFrame) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fleet = metrics[metrics["method"].isin(["descriptor_clustering", "fcgnn"])].copy()
    generated = []

    def _save(fig, name: str) -> None:
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        generated.append(name)

    if not fleet.empty and "campaign_precision" in fleet.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        for cfg, g in fleet.groupby("framework_config"):
            ax.scatter(g["campaign_recall"], g["campaign_precision"], label=cfg, alpha=0.6)
        ax.set_xlabel("Campaign recall")
        ax.set_ylabel("Campaign precision")
        ax.legend()
        ax.set_title("Campaign precision vs recall")
        _save(fig, "figure_MDF1_campaign_precision_recall")

        agg = fleet.groupby(["framework_config", "diversity_level"])["campaign_f1"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        for cfg, g in agg.groupby("framework_config"):
            ax.plot(g["diversity_level"], g["campaign_f1"], marker="o", label=cfg)
        ax.set_xlabel("Diversity level")
        ax.set_ylabel("Campaign F1")
        ax.legend()
        _save(fig, "figure_MDF2_campaign_F1_vs_diversity")

        if "false_campaign_alert_rate" in fleet.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            fleet.groupby("framework_config")["false_campaign_alert_rate"].mean().plot(kind="bar", ax=ax)
            ax.set_ylabel("False campaign alert rate")
            ax.set_title("False campaign rate by configuration")
            _save(fig, "figure_MDF4_false_campaign_rate")

        if "benign_vehicles_included" in fleet.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            fleet.groupby("framework_config")["benign_vehicles_included"].mean().plot(kind="bar", ax=ax)
            ax.set_ylabel("Benign vehicles included")
            _save(fig, "figure_MDF5_benign_vehicles_included")

    return generated
