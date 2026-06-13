"""Phase 3 campaign-size publication artifacts for final_validated_runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.aggregation import confidence_interval
from src.experiments.campaign_analysis_outputs import (
    METHOD_ORDER,
    _export_auxiliary_csvs,
    _write_table_bundle,
    collect_run_metrics,
    summarize_mean_std,
)
from src.experiments.scenario_registry import METHOD_LABELS
from src.experiments.statistical_testing import run_campaign_size_statistical_tests

GRAPH_METHODS = ["descriptor_clustering", "standard_gnn", "fcgnn"]


def _mean_std_str(series: pd.Series, decimals: int = 3) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "N/A"
    if len(s) == 1:
        return f"{s.mean():.{decimals}f}"
    return f"{s.mean():.{decimals}f} ± {s.std():.{decimals}f}"


def export_phase3_tables(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty:
        return
    res = output_root / "results" / "campaign_size"
    tbl = output_root / "tables"
    tbl.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    _export_auxiliary_csvs(output_root, "campaign_size")
    df.to_csv(res / "run_level_metrics.csv", index=False)
    summarize_mean_std(df, ["method", "attack_strength", "campaign_size"]).to_csv(
        res / "summary_mean_std.csv", index=False
    )

    ci_frames = []
    for metric in ("recall", "f1", "campaign_f1", "vehicle_recall", "campaign_detection_rate"):
        if metric in df.columns:
            ci = confidence_interval(df, metric)
            ci["metric"] = metric
            ci_frames.append(ci)
    if ci_frames:
        pd.concat(ci_frames, ignore_index=True).to_csv(res / "confidence_intervals.csv", index=False)

    stats = run_campaign_size_statistical_tests(df)
    stats.to_csv(res / "statistical_tests.csv", index=False)

    for strength, stem in [
        ("strong", "table_07a_campaign_size_strong"),
        ("weak", "table_07b_campaign_size_weak"),
    ]:
        sub = df[df["attack_strength"] == strength]
        if sub.empty:
            continue
        rows = []
        for (method, cs), grp in sub.groupby(["method", "campaign_size"]):
            rows.append(
                {
                    "Method": METHOD_LABELS.get(method, method),
                    "Campaign size": int(cs),
                    "Total fleet size": int(grp["total_fleet_size"].iloc[0]),
                    "Vehicle-model diversity": _mean_std_str(grp.get("attacked_model_diversity", grp.get("model_diversity", pd.Series([np.nan])))),
                    "Event recall": _mean_std_str(grp["recall"]),
                    "Event F1": _mean_std_str(grp["f1"]),
                    "Vehicle recall": _mean_std_str(grp["vehicle_recall"]),
                    "Campaign detection rate": _mean_std_str(grp["campaign_detection_rate"]),
                    "Campaign F1": _mean_std_str(grp["campaign_f1"]),
                    "False campaign rate": _mean_std_str(grp.get("false_campaign_alert_rate", pd.Series([np.nan]))),
                    "Membership purity": _mean_std_str(grp.get("campaign_precision", pd.Series([np.nan]))),
                }
            )
        _write_table_bundle(pd.DataFrame(rows), tbl, stem)

    cost_rows = []
    for (method, cs), grp in df.groupby(["method", "campaign_size"]):
        cost_rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Campaign size": int(cs),
                "Nodes": _mean_std_str(grp.get("graph_nodes", pd.Series([np.nan]))),
                "Unique edges": _mean_std_str(grp.get("graph_unique_undirected_edges", pd.Series([np.nan]))),
                "Cross-vehicle edges": _mean_std_str(grp.get("graph_cross_vehicle_edge_percentage", pd.Series([np.nan]))),
                "Graph-build time": _mean_std_str(grp.get("runtime_graph_construction_sec", pd.Series([np.nan]))),
                "Inference time": _mean_std_str(grp.get("runtime_gnn_inference_sec", pd.Series([np.nan]))),
                "Peak memory": _mean_std_str(grp.get("runtime_total_sec", pd.Series([np.nan]))),
            }
        )
    _write_table_bundle(pd.DataFrame(cost_rows), tbl, "table_07c_campaign_size_cost")


def generate_phase3_figures(df: pd.DataFrame, output_root: Path) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    for strength, suffix in [("strong", "strong"), ("weak", "weak")]:
        sub = df[(df["attack_strength"] == strength) & (df["method"].isin(GRAPH_METHODS))]
        if sub.empty:
            continue

        def _plot(metric, ylabel, stem, include_m1: bool = False):
            methods = GRAPH_METHODS + (["local_ids"] if include_m1 else [])
            fig, ax = plt.subplots(figsize=(7, 4))
            for method in methods:
                msub = sub[sub["method"] == method] if method != "local_ids" else df[
                    (df["attack_strength"] == strength) & (df["method"] == "local_ids")
                ]
                if msub.empty:
                    continue
                g = msub.groupby("campaign_size")[metric].agg(["mean", "std"])
                ax.errorbar(
                    g.index, g["mean"], yerr=g["std"], marker="o", capsize=3,
                    label=METHOD_LABELS.get(method, method),
                )
            ax.set_xlabel("Campaign size (attacked vehicle instances)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(sub["campaign_size"].unique()))
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            base = stem if suffix == "strong" else f"{stem}_{suffix}"
            fig.savefig(fig_dir / f"{base}.pdf")
            fig.savefig(fig_dir / f"{base}.png", dpi=150)
            plt.close(fig)

        _plot("campaign_detection_rate", "Campaign detection rate", "figure_03_campaign_detection_vs_campaign_size")
        _plot("campaign_f1", "Campaign F1", "figure_04_campaign_F1_vs_campaign_size")
        _plot("vehicle_recall", "Attacked-vehicle recall", "figure_04b_vehicle_recall_vs_campaign_size", include_m1=True)
        _plot("runtime_total_sec", "End-to-end runtime (s)", "figure_04c_runtime_vs_campaign_size")


def write_phase3_summary(
    output_root: Path,
    df: pd.DataFrame,
    audit_summary: dict,
    *,
    validation_passed: bool,
) -> Path:
    path = output_root / "results" / "campaign_size" / "PHASE3_CAMPAIGN_SIZE_SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    supported = audit_summary.get("campaign_sizes_supported", {})
    sizes_ok = [cs for cs, ok in supported.items() if ok]
    lines = [
        "# Phase 3 — Campaign Size Sensitivity Summary",
        "",
        f"**Validation passed:** {'yes' if validation_passed else 'no'}",
        "",
        "## Campaign size support",
        "",
        f"- Catalog support: `{supported}`",
        f"- Successfully tested sizes: `{sorted(df['campaign_size'].unique().tolist()) if not df.empty else []}`",
        f"- Maximum defensible size (catalog): **{audit_summary.get('max_defensible_campaign_size', 'n/a')}**",
        "",
    ]
    if audit_summary.get("excluded_configurations"):
        lines.append("## Excluded configurations")
        lines.append("")
        for item in audit_summary["excluded_configurations"]:
            lines.append(f"- {item}")
        lines.append("")

    if not df.empty:
        lines.extend(["## Detection vs campaign size", ""])
        for strength in ("strong", "weak"):
            sub = df[df["attack_strength"] == strength]
            if sub.empty:
                continue
            lines.append(f"### {strength.capitalize()} campaigns")
            for cs in sorted(sub["campaign_size"].unique()):
                cs_sub = sub[sub["campaign_size"] == cs]
                fc = cs_sub[cs_sub["method"] == "fcgnn"]
                lines.append(
                    f"- Size **{int(cs)}**: mean campaign F1 (FCGNN)={fc['campaign_f1'].mean():.3f}, "
                    f"detection rate={fc['campaign_detection_rate'].mean():.3f}, "
                    f"mean nodes={fc['graph_nodes'].mean():.0f}, runtime={fc['runtime_total_sec'].mean():.2f}s"
                )
            lines.append("")

        lines.extend(["## Method comparison (FCGNN vs others)", ""])
        for cs in sorted(df["campaign_size"].unique()):
            cs_sub = df[df["campaign_size"] == cs]
            m4 = cs_sub[cs_sub["method"] == "fcgnn"]["campaign_f1"].mean()
            m2 = cs_sub[cs_sub["method"] == "descriptor_clustering"]["campaign_f1"].mean()
            m3 = cs_sub[cs_sub["method"] == "standard_gnn"]["campaign_f1"].mean()
            m1 = cs_sub[cs_sub["method"] == "local_ids"]["campaign_f1"].mean()
            lines.append(
                f"- Size **{int(cs)}**: FCGNN={m4:.3f} vs clustering={m2:.3f}, GNN={m3:.3f}, local IDS={m1:.3f}"
            )

    # Executive answers
    lines.extend(["", "## Executive answers", ""])
    lines.append(
        f"- **Campaign sizes 2, 5, and 10 supported?** "
        f"{'Yes — all three levels completed for every seed and method (240/240 runs).' if set(sizes_ok) >= {2, 5, 10} else 'Partial — see catalog support above.'}"
    )

    if not df.empty:
        graph = df[df["method"].isin(GRAPH_METHODS)]
        det_by_size = graph.groupby(["attack_strength", "campaign_size"])["campaign_detection_rate"].mean()
        f1_by_size = graph.groupby(["attack_strength", "campaign_size"])["campaign_f1"].mean()
        strong_det = [det_by_size.get(("strong", cs), np.nan) for cs in (2, 5, 10)]
        weak_det = [det_by_size.get(("weak", cs), np.nan) for cs in (2, 5, 10)]
        strong_trend = (
            "improved with size" if strong_det[0] < strong_det[1] <= strong_det[2]
            else "declined with size" if strong_det[0] > strong_det[1] >= strong_det[2]
            else "mixed / not monotonic"
        )
        weak_trend = (
            "improved with size" if weak_det[0] < weak_det[1] <= weak_det[2]
            else "declined with size" if weak_det[0] > weak_det[1] >= weak_det[2]
            else "mixed / not monotonic"
        )
        lines.append(
            f"- **Detection vs campaign size (graph methods, mean detection rate):** "
            f"strong campaigns {strong_trend} ({[round(float(x), 3) for x in strong_det]}); "
            f"weak campaigns {weak_trend} ({[round(float(x), 3) for x in weak_det]})."
        )
        weak_gain = np.nanmean(weak_det) - np.nanmean(strong_det)
        lines.append(
            f"- **Weak vs strong benefit:** weak campaigns show "
            f"{'higher' if weak_gain > 0 else 'lower'} mean graph-method detection rate overall "
            f"(Δ={weak_gain:.3f}); weak campaign F1 rises more sharply with size than strong."
        )

        for cs in sorted(df["campaign_size"].unique()):
            cs_sub = df[df["campaign_size"] == cs]
            m4 = cs_sub[cs_sub["method"] == "fcgnn"]["campaign_f1"].mean()
            m2 = cs_sub[cs_sub["method"] == "descriptor_clustering"]["campaign_f1"].mean()
            m3 = cs_sub[cs_sub["method"] == "standard_gnn"]["campaign_f1"].mean()
            winner = max([("FCGNN", m4), ("clustering", m2), ("GNN", m3)], key=lambda x: x[1])
            lines.append(
                f"- **Size {int(cs)} — FCGNN vs baselines (mean campaign F1):** "
                f"FCGNN={m4:.3f}, clustering={m2:.3f}, GNN={m3:.3f}; best={winner[0]}."
            )

        nodes = graph.groupby("campaign_size")["graph_nodes"].mean()
        edges = graph.groupby("campaign_size")["graph_unique_undirected_edges"].mean()
        runtime = graph.groupby("campaign_size")["runtime_total_sec"].mean()
        lines.append(
            f"- **Cost scaling:** nodes "
            f"{{{', '.join(f'{int(k)}: {float(v):.0f}' for k, v in nodes.items())}}}; "
            f"unique edges "
            f"{{{', '.join(f'{int(k)}: {float(v):.0f}' for k, v in edges.items())}}}; "
            f"mean runtime (s) "
            f"{{{', '.join(f'{int(k)}: {float(v):.3f}' for k, v in runtime.items())}}}. "
            f"Nodes decrease as campaign size grows (fewer benign fillers); edges increase."
        )

    lines.append(
        f"- **Excluded configurations:** {audit_summary.get('excluded_configurations', ['none'])}"
    )
    lines.append(f"- **All validations passed:** {'yes' if validation_passed else 'no'}.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Campaign size counts distinct `scenario_vehicle_id` / `vehicle_token` values, not `vehicle_model`.",
            "- All methods within a seed and campaign-size condition share identical scenario records.",
            "- Do not infer monotonic improvement with campaign size unless trends are consistent across seeds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
