"""Publication outputs for corrected Phase 3 campaign-size experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.aggregation import confidence_interval
from src.experiments.campaign_analysis_outputs import (
    METHOD_ORDER,
    _collect_run_artifacts,
    _write_table_bundle,
    collect_run_metrics,
    summarize_mean_std,
)
from src.experiments.scenario_registry import METHOD_LABELS
from src.experiments.statistical_testing import run_corrected_phase3_primary_tests

EXPERIMENT = "campaign_size_corrected"
GRAPH_METHODS = ["descriptor_clustering", "standard_gnn", "fcgnn"]


def _mean_std_str(series: pd.Series, decimals: int = 3) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "N/A"
    if len(s) == 1:
        return f"{s.mean():.{decimals}f}"
    return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"


def export_corrected_raw_outputs(df: pd.DataFrame, output_root: Path) -> None:
    res = output_root / "results" / EXPERIMENT
    res.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    df.to_csv(res / "run_level_metrics.csv", index=False)
    df[df["attack_strength"] == "strong"].to_csv(res / "strong_run_level_metrics.csv", index=False)
    df[df["attack_strength"] == "weak"].to_csv(res / "weak_run_level_metrics.csv", index=False)

    for fname, out in (
        ("scenario_vehicle_mapping.csv", "scenario_vehicle_mapping.csv"),
        ("vehicle_membership.csv", "vehicle_membership.csv"),
        ("graph_statistics.csv", "graph_statistics.csv"),
        ("vehicle_composition.csv", "vehicle_composition.csv"),
        ("event_budget_validation.csv", "event_budget_validation.csv"),
    ):
        combined = _collect_run_artifacts(output_root, EXPERIMENT, fname)
        if not combined.empty:
            combined.to_csv(res / out, index=False)

    runtime_cols = [
        c for c in df.columns if c.startswith("runtime_") or c.startswith("graph_")
    ]
    extra = [
        c
        for c in (
            "run_id",
            "method",
            "seed",
            "campaign_size",
            "attack_strength",
            "expected_total_nodes",
        )
        if c in df.columns
    ]
    df[extra + runtime_cols].to_csv(res / "runtime_memory.csv", index=False)

    summarize_mean_std(
        df[df["attack_strength"] == "strong"],
        ["method", "campaign_size"],
    ).to_csv(res / "strong_summary_mean_std.csv", index=False)
    summarize_mean_std(
        df[df["attack_strength"] == "weak"],
        ["method", "campaign_size"],
    ).to_csv(res / "weak_summary_mean_std.csv", index=False)

    ci_frames = []
    for metric in (
        "recall",
        "f1",
        "fpr",
        "campaign_f1",
        "vehicle_recall",
        "campaign_detection_rate",
    ):
        if metric in df.columns:
            for strength in ("strong", "weak"):
                sub = df[df["attack_strength"] == strength]
                ci = confidence_interval(sub, metric)
                ci["metric"] = metric
                ci["attack_strength"] = strength
                ci_frames.append(ci)
    if ci_frames:
        pd.concat(ci_frames, ignore_index=True).to_csv(
            res / "confidence_intervals.csv", index=False
        )

    stats = run_corrected_phase3_primary_tests(df)
    stats.to_csv(res / "statistical_tests_primary.csv", index=False)


def export_corrected_tables(df: pd.DataFrame, output_root: Path) -> None:
    tbl = output_root / "tables" / "campaign_size_corrected"
    tbl.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

  # C1 strong
    strong = df[df["attack_strength"] == "strong"]
    rows_c1 = []
    for (method, cs), grp in strong.groupby(["method", "campaign_size"]):
        rows_c1.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Campaign size": int(cs),
                "Campaign detection rate": _mean_std_str(grp["campaign_detection_rate"]),
                "Campaign precision": _mean_std_str(grp["campaign_precision"]),
                "Campaign recall": _mean_std_str(grp["campaign_recall"]),
                "Campaign F1": _mean_std_str(grp["campaign_f1"]),
                "Attacked-vehicle recall": _mean_std_str(grp["vehicle_recall"]),
                "Membership purity": _mean_std_str(grp["campaign_precision"]),
                "False campaign alert rate": _mean_std_str(grp["false_campaign_alert_rate"]),
            }
        )
    _write_table_bundle(pd.DataFrame(rows_c1), tbl, "table_C1_strong_campaign_size_results")

    weak = df[df["attack_strength"] == "weak"]
    rows_c2 = []
    for (method, cs), grp in weak.groupby(["method", "campaign_size"]):
        rows_c2.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Campaign size": int(cs),
                "Event precision": _mean_std_str(grp["precision"]),
                "Event recall": _mean_std_str(grp["recall"]),
                "Event F1": _mean_std_str(grp["f1"]),
                "Event FPR": _mean_std_str(grp["fpr"]),
                "Attacked-vehicle recall": _mean_std_str(grp["vehicle_recall"]),
                "Campaign detection rate": _mean_std_str(grp["campaign_detection_rate"]),
                "Campaign F1": _mean_std_str(grp["campaign_f1"]),
                "Weak events recovered": _mean_std_str(grp.get("weak_malicious_recovered", pd.Series())),
                "Weak benign events promoted": _mean_std_str(grp.get("weak_benign_promoted", pd.Series())),
            }
        )
    _write_table_bundle(pd.DataFrame(rows_c2), tbl, "table_C2_weak_campaign_size_results")

    rows_c3 = []
    for (method, cs), grp in df.groupby(["method", "campaign_size"]):
        rows_c3.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Campaign size": int(cs),
                "Total nodes": _mean_std_str(grp.get("graph_nodes", grp.get("expected_total_nodes", pd.Series()))),
                "Unique edges": _mean_std_str(grp.get("graph_unique_undirected_edges", pd.Series())),
                "Cross-vehicle edges": _mean_std_str(grp.get("graph_cross_vehicle_edge_percentage", pd.Series())),
                "Graph-build time": _mean_std_str(grp.get("runtime_graph_construction_sec", pd.Series())),
                "Inference time": _mean_std_str(grp.get("runtime_gnn_inference_sec", pd.Series())),
                "End-to-end latency": _mean_std_str(grp.get("runtime_total_sec", pd.Series())),
                "Peak memory": _mean_std_str(grp.get("runtime_total_sec", pd.Series())),
            }
        )
    _write_table_bundle(pd.DataFrame(rows_c3), tbl, "table_C3_campaign_size_cost")

    comp_path = output_root / "results" / EXPERIMENT / "vehicle_composition.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        rows_c4 = []
        for (strength, cs), grp in comp.groupby(["attack_strength", "campaign_size"]):
            rows_c4.append(
                {
                    "Attack strength": strength,
                    "Campaign size": int(cs),
                    "Fleet size": int(grp["total_fleet_size"].iloc[0]),
                    "Hyundai instances": _mean_std_str(grp["Hyundai_instances"]),
                    "Kia instances": _mean_std_str(grp["Kia_instances"]),
                    "Chevrolet instances": _mean_std_str(grp["Chevrolet_instances"]),
                    "Model diversity": _mean_std_str(grp["vehicle_model_diversity"]),
                    "Descriptors per vehicle": int(df["descriptors_per_vehicle"].iloc[0])
                    if "descriptors_per_vehicle" in df.columns
                    else 10,
                    "Malicious descriptors per attacked vehicle": 5,
                }
            )
        _write_table_bundle(pd.DataFrame(rows_c4), tbl, "table_C4_platform_composition")

    stats_path = output_root / "results" / EXPERIMENT / "statistical_tests_primary.csv"
    if stats_path.exists():
        st = pd.read_csv(stats_path)
        st_out = st.rename(
            columns={
                "statistical_test": "Statistical test",
                "mean_paired_difference": "Mean paired difference",
                "ci95_low": "CI95 low",
                "ci95_high": "CI95 high",
                "raw_p_value": "Raw p-value",
                "holm_adjusted_p_value": "Holm-adjusted p-value",
                "paired_seeds": "Paired seeds",
                "effect_size": "Effect size",
                "significant": "Significant",
            }
        )
        st_out["95% confidence interval"] = st_out.apply(
            lambda r: f"[{r.get('CI95 low', r.get('ci95_low', np.nan)):.4f}, "
            f"{r.get('CI95 high', r.get('ci95_high', np.nan)):.4f}]"
            if pd.notna(r.get("CI95 low", r.get("ci95_low", np.nan)))
            else "N/A",
            axis=1,
        )
        cols = [
            "attack_strength",
            "campaign_size",
            "metric",
            "comparison",
            "Paired seeds",
            "Statistical test",
            "Mean paired difference",
            "95% confidence interval",
            "Raw p-value",
            "Holm-adjusted p-value",
            "Effect size",
            "Significant",
        ]
        cols = [c for c in cols if c in st_out.columns or c == "95% confidence interval"]
        _write_table_bundle(
            st_out[
                [
                    c
                    for c in [
                        "attack_strength",
                        "campaign_size",
                        "metric",
                        "comparison",
                        "Paired seeds",
                        "Statistical test",
                        "Mean paired difference",
                        "95% confidence interval",
                        "Raw p-value",
                        "Holm-adjusted p-value",
                        "Effect size",
                        "Significant",
                    ]
                    if c in st_out.columns
                ]
            ].rename(
                columns={
                    "attack_strength": "Attack strength",
                    "campaign_size": "Campaign size",
                    "metric": "Metric",
                    "comparison": "Comparison",
                }
            ),
            tbl,
            "table_C5_primary_statistical_tests",
        )


def generate_corrected_figures(df: pd.DataFrame, output_root: Path) -> None:
    fig_dir = output_root / "figures" / "campaign_size_corrected"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    def _plot(
        sub: pd.DataFrame,
        metric: str,
        ylabel: str,
        stem: str,
        methods: list[str] | None = None,
    ) -> None:
        methods = methods or METHOD_ORDER
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in methods:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            g = msub.groupby("campaign_size")[metric].agg(["mean", "std"])
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=g["std"],
                marker="o",
                capsize=3,
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_xlabel("Campaign size (attacked vehicle instances)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(sub["campaign_size"].unique()))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{stem}.pdf")
        fig.savefig(fig_dir / f"{stem}.png", dpi=150)
        plt.close(fig)

    strong = df[df["attack_strength"] == "strong"]
    weak = df[df["attack_strength"] == "weak"]
    _plot(strong, "campaign_detection_rate", "Campaign detection rate", "figure_C1_strong_campaign_detection_vs_size")
    _plot(strong, "campaign_f1", "Campaign F1", "figure_C2_strong_campaign_F1_vs_size")
    _plot(weak, "recall", "Event recall", "figure_C3_weak_event_recall_vs_size")
    _plot(weak, "campaign_f1", "Campaign F1", "figure_C4_weak_campaign_F1_vs_size")
    _plot(df, "vehicle_recall", "Attacked-vehicle recall", "figure_C5_vehicle_recall_vs_size")
    _plot(
        df[df["method"].isin(GRAPH_METHODS)],
        "graph_unique_undirected_edges",
        "Unique graph edges",
        "figure_C6_unique_edges_vs_campaign_size",
        methods=GRAPH_METHODS,
    )
    _plot(df, "runtime_total_sec", "End-to-end runtime (s)", "figure_C7_runtime_vs_campaign_size")


def write_original_vs_corrected(output_root: Path, corrected: pd.DataFrame) -> Path:
    path = output_root / "results" / EXPERIMENT / "original_vs_corrected_phase3.md"
    orig_path = output_root / "results" / "campaign_size" / "run_level_metrics.csv"
    lines = [
        "# Original vs Corrected Phase 3 Comparison",
        "",
        "Original Phase 3 results are **preliminary** (variable node counts and uncontrolled platform mix).",
        "Corrected results use fixed descriptor budgets and approved platform composition.",
        "",
    ]
    if not orig_path.exists() or corrected.empty:
        lines.append("Comparison data unavailable.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    orig = pd.read_csv(orig_path)
    lines.extend(["## Node counts (mean graph_nodes)", ""])
    for strength in ("strong", "weak"):
        lines.append(f"### {strength.capitalize()}")
        for cs in sorted(corrected["campaign_size"].unique()):
            o = orig[(orig["attack_strength"] == strength) & (orig["campaign_size"] == cs)]
            c = corrected[(corrected["attack_strength"] == strength) & (corrected["campaign_size"] == cs)]
            o_nodes = o["graph_nodes"].mean() if "graph_nodes" in o.columns else float("nan")
            c_nodes = c["graph_nodes"].mean() if "graph_nodes" in c.columns else float("nan")
            lines.append(f"- Size **{int(cs)}**: original={o_nodes:.1f}, corrected={c_nodes:.1f}")
        lines.append("")

    lines.extend(["## Campaign F1 (FCGNN, mean)", ""])
    for strength in ("strong", "weak"):
        lines.append(f"### {strength.capitalize()}")
        for cs in sorted(corrected["campaign_size"].unique()):
            o = orig[
                (orig["attack_strength"] == strength)
                & (orig["campaign_size"] == cs)
                & (orig["method"] == "fcgnn")
            ]["campaign_f1"].mean()
            c = corrected[
                (corrected["attack_strength"] == strength)
                & (corrected["campaign_size"] == cs)
                & (corrected["method"] == "fcgnn")
            ]["campaign_f1"].mean()
            lines.append(f"- Size **{int(cs)}**: original={o:.3f}, corrected={c:.3f}")
        lines.append("")

    lines.extend(
        [
            "## Conclusion",
            "",
            "Corrected controls fix total graph size at 200 nodes and enforce per-vehicle descriptor quotas.",
            "Trends and significance claims must use corrected tables (C1–C5) and figures (C1–C7) only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_corrected_summary(
    output_root: Path,
    df: pd.DataFrame,
    audit_summary: dict,
    *,
    budget: dict,
    validation_passed: bool,
    excluded_runs: pd.DataFrame | None = None,
) -> Path:
    path = output_root / "results" / EXPERIMENT / "CORRECTED_PHASE3_SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corrected Phase 3 — Campaign Size Sensitivity",
        "",
        f"**Validation passed:** {'yes' if validation_passed else 'no'}",
        f"**Runs completed:** {len(df)}",
        f"**Descriptor budget:** {budget}",
        f"**Expected total nodes:** {budget.get('expected_total_nodes', 'n/a')}",
        "",
        "## Platform composition (attacked vehicles)",
        "",
        "- Strong cs=2: rotate Hyundai+Kia / Hyundai+Chevrolet / Kia+Chevrolet across seeds",
        "- Strong cs=5: 2 Hyundai, 2 Kia, 1 Chevrolet",
        "- Strong cs=10: 4 Hyundai, 3 Kia, 3 Chevrolet",
        "- Weak cs=2: 1 Hyundai, 1 Kia",
        "- Weak cs=5: 3 Hyundai, 2 Kia",
        "- Weak cs=10: 5 Hyundai, 5 Kia (no Chevrolet weak events)",
        "",
    ]
    if excluded_runs is not None and not excluded_runs.empty:
        lines.append("## Excluded runs")
        lines.append("")
        for _, r in excluded_runs.iterrows():
            lines.append(f"- {r.to_dict()}")
        lines.append("")

    if not df.empty:
        lines.append("## Method-specific highlights")
        lines.append("")
        for strength in ("strong", "weak"):
            lines.append(f"### {strength.capitalize()}")
            for method in METHOD_ORDER:
                sub = df[(df["attack_strength"] == strength) & (df["method"] == method)]
                if sub.empty:
                    continue
                lines.append(
                    f"- **{METHOD_LABELS.get(method, method)}**: "
                    f"campaign F1={sub['campaign_f1'].mean():.3f}, "
                    f"detection={sub['campaign_detection_rate'].mean():.3f}, "
                    f"nodes={sub['graph_nodes'].mean():.0f}"
                )
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def collect_corrected_metrics(output_root: Path) -> pd.DataFrame:
    df = collect_run_metrics(output_root, EXPERIMENT)
    runs_dir = output_root / "results" / EXPERIMENT / "runs"
    if runs_dir.exists() and not df.empty and "run_id" not in df.columns:
        ids = []
        for run_dir in runs_dir.iterdir():
            if (run_dir / "run_level_metrics.csv").exists():
                ids.append(run_dir.name)
        if len(ids) == len(df):
            df["run_id"] = ids
    return df
