"""Aggregate campaign-analysis results into tables and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.scenario_registry import METHOD_LABELS
from src.experiments.statistical_testing import run_paired_comparisons

METHOD_ORDER = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]
GRAPH_METHODS = ["descriptor_clustering", "standard_gnn", "fcgnn"]


def _diversity_column(df: pd.DataFrame) -> str:
    for col in ("diversity_level", "attacked_model_diversity", "model_diversity"):
        if col in df.columns and df[col].notna().any():
            return col
    return "diversity_level"


def _collect_run_artifacts(output_root: Path, experiment: str, filename: str) -> pd.DataFrame:
    runs_dir = output_root / "results" / experiment / "runs"
    frames: list[pd.DataFrame] = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        p = run_dir / filename
        if p.exists():
            df = pd.read_csv(p)
            df["run_id"] = run_dir.name
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_run_metrics(output_root: Path, experiment: str) -> pd.DataFrame:
    runs_dir = output_root / "results" / experiment / "runs"
    frames: list[pd.DataFrame] = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        p = run_dir / "run_level_metrics.csv"
        if p.exists():
            row = pd.read_csv(p)
            row["run_id"] = run_dir.name
            frames.append(row)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    keys = ["experiment", "attack_strength", "method", "seed"]
    if experiment == "model_diversity":
        keys.append(_diversity_column(df))
    else:
        keys.append("campaign_size")
    keys = [k for k in keys if k in df.columns]
    return df.drop_duplicates(subset=keys, keep="last")


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(group_cols)[numeric].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg.reset_index()


def _write_table_bundle(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    md = ["| " + " | ".join(df.columns.astype(str)) + " |",
          "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        md.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex_lines = [
        "\\begin{tabular}{" + "l" * len(df.columns) + "}",
        " \\toprule",
        " ".join(df.columns.astype(str)) + " \\\\",
        " \\midrule",
    ]
    for _, row in df.iterrows():
        tex_lines.append(" ".join(str(row[c]) for c in df.columns) + " \\\\")
    tex_lines.extend([" \\bottomrule", "\\end{tabular}"])
    (out_dir / f"{stem}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def _export_auxiliary_csvs(output_root: Path, experiment: str) -> None:
    res = output_root / "results" / experiment
    res.mkdir(parents=True, exist_ok=True)
    for fname, out_name in [
        ("vehicle_membership.csv", "vehicle_membership.csv"),
        ("scenario_vehicle_mapping.csv", "scenario_vehicle_mapping.csv"),
        ("graph_statistics.csv", "graph_statistics.csv"),
        ("descriptor_similarity.csv", "descriptor_similarity.csv"),
        ("runtime_memory.json", None),
    ]:
        if fname.endswith(".json"):
            continue
        combined = _collect_run_artifacts(output_root, experiment, fname)
        if not combined.empty:
            combined.to_csv(res / out_name, index=False)
    metrics = collect_run_metrics(output_root, experiment)
    if not metrics.empty:
        runtime_cols = [c for c in metrics.columns if c.startswith("runtime_") or c.startswith("graph_")]
        extra = [c for c in ("method", "seed", "campaign_size", "attack_strength", "model_diversity") if c in metrics.columns]
        metrics[extra + runtime_cols].to_csv(res / "runtime_memory.csv", index=False)


def export_experiment_a_tables(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty:
        return
    res = output_root / "results" / "campaign_size"
    tbl = output_root / "tables" / "campaign_size"
    res.mkdir(parents=True, exist_ok=True)
    _export_auxiliary_csvs(output_root, "campaign_size")
    df.to_csv(res / "run_level_metrics.csv", index=False)
    summarize_mean_std(df, ["method", "attack_strength", "campaign_size"]).to_csv(
        res / "summary_mean_std.csv", index=False
    )
    run_paired_comparisons(df, scenario_key="campaign_size").to_csv(
        res / "statistical_tests.csv", index=False
    )

    for strength, stem in [("strong", "table_A1_campaign_size_strong"), ("weak", "table_A2_campaign_size_weak")]:
        sub = df[df["attack_strength"] == strength]
        if sub.empty:
            continue
        diversity_col = "model_diversity" if "model_diversity" in sub.columns else "attacked_model_diversity"
        agg_spec = {
            "total_fleet_size": ("total_fleet_size", "first"),
            "event_recall": ("recall", "mean"),
            "event_f1": ("f1", "mean"),
            "vehicle_recall": ("vehicle_recall", "mean"),
            "campaign_detection_rate": ("campaign_detection_rate", "mean"),
            "campaign_f1": ("campaign_f1", "mean"),
            "false_campaign_rate": ("false_campaign_alert_rate", "mean"),
            "membership_purity": ("campaign_precision", "mean"),
        }
        if diversity_col in sub.columns:
            agg_spec["model_diversity"] = (diversity_col, "mean")
        summary = sub.groupby(["method", "campaign_size"]).agg(**agg_spec).reset_index()
        summary["Method"] = summary["method"].map(METHOD_LABELS)
        summary = summary.rename(columns={"campaign_size": "Campaign size"})
        cols = [
            "Method", "Campaign size", "total_fleet_size", "model_diversity",
            "event_recall", "event_f1", "vehicle_recall", "campaign_detection_rate",
            "campaign_f1", "false_campaign_rate", "membership_purity",
        ]
        cols = [c for c in cols if c in summary.columns]
        _write_table_bundle(summary[cols], tbl, stem)

    cost = df.groupby(["method", "campaign_size"]).agg(
        nodes=("graph_nodes", "mean"),
        unique_edges=("graph_unique_undirected_edges", "mean"),
        cross_vehicle_edges=("graph_cross_vehicle_edge_percentage", "mean"),
        graph_build_time=("runtime_graph_construction_sec", "mean"),
        inference_time=("runtime_gnn_inference_sec", "mean"),
        peak_memory=("runtime_total_sec", "mean"),
    ).reset_index()
    cost["Method"] = cost["method"].map(METHOD_LABELS)
    cost = cost.rename(columns={"campaign_size": "Campaign size"})
    _write_table_bundle(
        cost[["Method", "Campaign size", "nodes", "unique_edges", "cross_vehicle_edges",
              "graph_build_time", "inference_time", "peak_memory"]],
        tbl,
        "table_A3_campaign_size_cost",
    )


def export_experiment_b_tables(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty:
        return
    res = output_root / "results" / "model_diversity"
    tbl = output_root / "tables" / "model_diversity"
    res.mkdir(parents=True, exist_ok=True)
    _export_auxiliary_csvs(output_root, "model_diversity")
    df.to_csv(res / "run_level_metrics.csv", index=False)
    div_col = _diversity_column(df)
    summarize_mean_std(df, ["method", "attack_strength", div_col]).to_csv(
        res / "summary_mean_std.csv", index=False
    )
    run_paired_comparisons(df, scenario_key="model_diversity").to_csv(
        res / "statistical_tests.csv", index=False
    )

    for strength, stem in [("strong", "table_B1_model_diversity_strong"), ("weak", "table_B2_model_diversity_weak")]:
        sub = df[df["attack_strength"] == strength]
        if sub.empty:
            continue
        div_col = _diversity_column(sub)
        summary = sub.groupby(["method", div_col]).agg(
            campaign_size=("campaign_size", "first"),
            event_recall=("recall", "mean"),
            event_f1=("f1", "mean"),
            vehicle_recall=("vehicle_recall", "mean"),
            campaign_detection_rate=("campaign_detection_rate", "mean"),
            campaign_f1=("campaign_f1", "mean"),
            membership_purity=("campaign_precision", "mean"),
            false_campaign_rate=("false_campaign_alert_rate", "mean"),
        ).reset_index()
        summary["Method"] = summary["method"].map(METHOD_LABELS)
        summary = summary.rename(columns={div_col: "Model diversity"})
        _write_table_bundle(
            summary[["Method", "campaign_size", "Model diversity", "event_recall", "event_f1",
                     "vehicle_recall", "campaign_detection_rate", "campaign_f1",
                     "membership_purity", "false_campaign_rate"]].rename(
                columns={"campaign_size": "Campaign size"}
            ),
            tbl,
            stem,
        )

    div_col = _diversity_column(df)
    sim = df.groupby(div_col).agg(
        within_model_attack_similarity=("within_model_attack_similarity", "mean"),
        cross_model_attack_similarity=("cross_model_attack_similarity", "mean"),
        benign_cross_model_similarity=("benign_cross_model_similarity", "mean"),
        cross_model_edge_percentage=("cross_model_edge_percentage", "mean"),
        campaign_purity=("campaign_precision", "mean"),
    ).reset_index()
    sim = sim.rename(columns={div_col: "Model diversity"})
    _write_table_bundle(sim, tbl, "table_B3_cross_model_similarity")


def generate_experiment_a_figures(df: pd.DataFrame, output_root: Path) -> None:
    fig_dir = output_root / "figures" / "campaign_size"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "method" not in df.columns:
        return
    sub = df[df["method"].isin(GRAPH_METHODS)]
    if sub.empty:
        return

    def _plot(metric, ylabel, stem):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in GRAPH_METHODS:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            g = msub.groupby("campaign_size")[metric].mean()
            ax.plot(g.index, g.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Number of attacked vehicle instances")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{stem}.pdf")
        fig.savefig(fig_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

    _plot("campaign_detection_rate", "Campaign detection rate", "figure_A1_campaign_detection_vs_campaign_size")
    _plot("campaign_f1", "Campaign F1", "figure_A2_campaign_f1_vs_campaign_size")
    _plot("runtime_total_sec", "End-to-end runtime (s)", "figure_A3_runtime_vs_campaign_size")


def generate_experiment_b_figures(df: pd.DataFrame, output_root: Path) -> None:
    fig_dir = output_root / "figures" / "model_diversity"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if df.empty or "method" not in df.columns:
        return
    sub = df[df["method"].isin(GRAPH_METHODS)]
    if sub.empty:
        return

    def _plot(metric, ylabel, stem, xlabel="Number of vehicle models in campaign"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in GRAPH_METHODS:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            div_col = _diversity_column(msub)
            g = msub.groupby(div_col)[metric].mean()
            ax.plot(g.index, g.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{stem}.pdf")
        fig.savefig(fig_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

    _plot("campaign_f1", "Campaign F1", "figure_B1_campaign_f1_vs_model_diversity")
    _plot("vehicle_recall", "Attacked-vehicle recall", "figure_B2_vehicle_recall_vs_model_diversity")

    fig, ax = plt.subplots(figsize=(7, 4))
    div_col = _diversity_column(sub)
    sim = sub.groupby(div_col).agg(
        cross=("cross_model_attack_similarity", "mean"),
        benign=("benign_cross_model_similarity", "mean"),
    )
    ax.plot(sim.index, sim["cross"], marker="o", label="Cross-model attack similarity")
    ax.plot(sim.index, sim["benign"], marker="s", label="Cross-model benign similarity")
    ax.set_xlabel("Number of vehicle models in campaign")
    ax.set_ylabel("Cosine similarity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "figure_B3_cross_model_similarity.pdf")
    fig.savefig(fig_dir / "figure_B3_cross_model_similarity.png", dpi=150)
    plt.close(fig)


def write_final_summary(
    output_root: Path,
    audit_summary: dict,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> Path:
    path = output_root / "FINAL_CAMPAIGN_ANALYSIS_SUMMARY.md"
    lines = [
        "# Final Campaign Analysis Summary",
        "",
        "## Data availability",
        "",
        f"- Catalogued disjoint vehicle instances: **{audit_summary.get('catalog_instances', 'n/a')}**",
        f"- Maximum defensible campaign size: **{audit_summary.get('max_defensible_campaign_size', 'n/a')}**",
        f"- Campaign sizes supported (catalog): `{audit_summary.get('campaign_sizes_supported', {})}`",
        "",
        "## Experiment A — Campaign size sensitivity",
        "",
    ]
    if not df_a.empty:
        for cs in sorted(df_a["campaign_size"].unique()):
            sub = df_a[df_a["campaign_size"] == cs]
            lines.append(
                f"- Campaign size **{cs}**: mean campaign F1 "
                f"FCGNN={sub[sub['method']=='fcgnn']['campaign_f1'].mean():.3f}, "
                f"clustering={sub[sub['method']=='descriptor_clustering']['campaign_f1'].mean():.3f}, "
                f"GNN={sub[sub['method']=='standard_gnn']['campaign_f1'].mean():.3f}; "
                f"mean runtime={sub['runtime_total_sec'].mean():.2f}s"
            )
    else:
        lines.append("- No runs completed yet.")
    lines.extend(["", "## Experiment B — Model diversity sensitivity", ""])
    if not df_b.empty:
        div_col = _diversity_column(df_b)
        for d in sorted(df_b[div_col].dropna().unique()):
            sub = df_b[df_b[div_col] == d]
            lines.append(
                f"- Diversity **{d}** models: mean campaign F1 "
                f"FCGNN={sub[sub['method']=='fcgnn']['campaign_f1'].mean():.3f}; "
                f"cross-model edge %={sub['cross_model_edge_percentage'].mean():.1f}"
            )
    else:
        lines.append("- No runs completed yet.")
    lines.extend(
        [
            "",
            "## Distinctions",
            "",
            "- **Campaign size** = distinct attacked `scenario_vehicle_id` count.",
            "- **Model diversity** = distinct `vehicle_model` values among attacked campaign members.",
            "- **Fleet size** = attacked + benign instance count (fixed at 20 where supported).",
            "- No descriptor rows were duplicated to simulate additional vehicles.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
