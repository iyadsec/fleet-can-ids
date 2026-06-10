"""Aggregate run outputs into scenario tables, figures, and summary."""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.scenario_registry import METHOD_LABELS, SCENARIO_REGISTRY
from src.experiments.statistical_testing import run_paired_comparisons

METHOD_ORDER = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]


def collect_run_metrics(output_root: Path, scenario_key: str) -> pd.DataFrame:
    runs_dir = output_root / "results" / scenario_key / "runs"
    frames: list[pd.DataFrame] = []
    if not runs_dir.exists():
        return pd.DataFrame()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "run_level_metrics.csv"
        if metrics_path.exists():
            frames.append(pd.read_csv(metrics_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_mean_std(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = ["method", "scenario_key", "campaign_size", "coordination_strength"]
    group_cols = [c for c in group_cols if c in df.columns]
    agg = df.groupby(group_cols)[numeric].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg.reset_index()


def confidence_interval(df: pd.DataFrame, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for keys, grp in df.groupby(["method", "scenario_key"]):
        vals = grp[metric].dropna().to_numpy()
        if len(vals) < 2:
            continue
        m, se = np.mean(vals), stats.sem(vals)
        h = se * stats.t.ppf(1 - alpha / 2, len(vals) - 1)
        rows.append({**dict(zip(["method", "scenario_key"], keys)), "mean": m, "ci95_low": m - h, "ci95_high": m + h})
    return pd.DataFrame(rows)


def export_scenario_tables(output_root: Path, scenario_key: str) -> None:
    df = collect_run_metrics(output_root, scenario_key)
    if df.empty:
        return
    res_dir = output_root / "results" / scenario_key
    tables_dir = output_root / "tables" / scenario_key
    res_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(res_dir / "run_level_metrics.csv", index=False)
    summary = summarize_mean_std(df)
    summary.to_csv(res_dir / "summary_mean_std.csv", index=False)
    stats_df = run_paired_comparisons(df, scenario_key=scenario_key)
    stats_df.to_csv(res_dir / "statistical_tests.csv", index=False)

    md_lines = [f"# {scenario_key}", "", "| " + " | ".join(summary.columns.astype(str)) + " |",
                "| " + " | ".join(["---"] * len(summary.columns)) + " |"]
    for _, row in summary.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in summary.columns) + " |")
    (tables_dir / f"table_{scenario_key}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    summary.to_csv(tables_dir / f"table_{scenario_key}.csv", index=False)


def generate_figures(output_root: Path) -> None:
    fig_root = output_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for key in SCENARIO_REGISTRY:
        df = collect_run_metrics(output_root, key)
        if not df.empty:
            all_metrics.append(df)
    if not all_metrics:
        return
    combined = pd.concat(all_metrics, ignore_index=True)

    # Fig 1: Campaign F1 vs coordination strength (S3/S4)
    sub = combined[combined["scenario_key"].isin(["S3_strong_campaign", "S4_weak_campaign"])]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            g = msub.groupby("coordination_strength")["campaign_f1"].mean()
            ax.plot(g.index, g.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Coordination strength")
        ax.set_ylabel("Campaign F1")
        ax.legend(fontsize=8)
        ax.set_title("Campaign F1 vs coordination strength")
        _save(fig, fig_root / "campaign_f1_vs_coordination_strength")

    # Fig 4/5: Method comparison S3/S4
    for scenario, stem in [("S3_strong_campaign", "method_comparison_S3"), ("S4_weak_campaign", "method_comparison_S4")]:
        s = combined[combined["scenario_key"] == scenario]
        if s.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, metric in zip(axes, ["recall", "f1", "campaign_f1"]):
            means = s.groupby("method")[metric].mean().reindex(METHOD_ORDER)
            ax.bar(range(len(means)), means.values)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in means.index], rotation=20, ha="right", fontsize=7)
            ax.set_title(metric)
        fig.suptitle(scenario)
        _save(fig, fig_root / stem)

    # Fig 6: Weak recovery
    s4 = combined[combined["scenario_key"] == "S4_weak_campaign"]
    if not s4.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        rec = s4.groupby("method")["weak_malicious_recovered"].mean().reindex(METHOD_ORDER)
        ax.bar(range(len(rec)), rec.values)
        ax.set_xticks(range(len(rec)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in rec.index], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Weak malicious events recovered (mean)")
        ax.set_title("S4 weak-event recovery by method")
        _save(fig, fig_root / "weak_event_recovery_by_method")

    # Fig 7: S4 recall improvement vs FPR cost
    if not s4.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        local = s4[s4["method"] == "local_ids"].groupby("seed")["recall"].mean()
        fcgnn = s4[s4["method"] == "fcgnn"].groupby("seed")["recall"].mean()
        fpr_local = s4[s4["method"] == "local_ids"].groupby("seed")["fpr"].mean()
        fpr_fcgnn = s4[s4["method"] == "fcgnn"].groupby("seed")["fpr"].mean()
        idx = local.index.intersection(fcgnn.index)
        ax.scatter(fpr_fcgnn.reindex(idx) - fpr_local.reindex(idx), fcgnn.reindex(idx) - local.reindex(idx))
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Δ FPR (FCGNN − Local)")
        ax.set_ylabel("Δ Recall (FCGNN − Local)")
        ax.set_title("S4 recall improvement vs FPR change")
        _save(fig, fig_root / "S4_recall_improvement_vs_fpr")


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def write_final_summary(output_root: Path, config_path: str) -> Path:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    lines = [
        "# Final Scenario Results Summary",
        "",
        f"**Git commit:** `{commit}`",
        f"**Config:** `{config_path}`",
        "",
        "## Scenarios executed",
        "",
    ]
    for key in SCENARIO_REGISTRY:
        df = collect_run_metrics(output_root, key)
        lines.append(f"- **{key}**: {len(df)} runs")

    s4 = collect_run_metrics(output_root, "S4_weak_campaign")
    if not s4.empty:
        lines.extend(["", "## Primary S4 results (FCGNN vs Local IDS)", ""])
        for metric in ["recall", "f1", "fpr", "weak_malicious_recovered", "campaign_detection_rate"]:
            local_m = s4[s4["method"] == "local_ids"][metric].mean()
            fcg_m = s4[s4["method"] == "fcgnn"][metric].mean()
            lines.append(f"- Δ {metric}: {fcg_m - local_m:+.4f} (FCGNN {fcg_m:.4f} vs Local {local_m:.4f})")

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"python scripts/run_new_scenario_experiments.py --config {config_path} --all-scenarios",
            "```",
            "",
        ]
    )
    path = output_root / "FINAL_SCENARIO_RESULTS_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
