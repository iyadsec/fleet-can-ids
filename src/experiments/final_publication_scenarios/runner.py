"""Orchestrate final publication scenario package."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.experiments.final_publication_scenarios.collect import collect_campaign_size_results, collect_scenario_results
from src.experiments.final_publication_scenarios.edge_sensitivity import run_edge_sensitivity
from src.experiments.final_publication_scenarios.figures import generate_all_figures
from src.experiments.final_publication_scenarios.guard import PublicationScenariosGuard
from src.experiments.final_publication_scenarios.inventory import (
    REQUIRED_SEEDS,
    build_source_inventory,
    write_source_selection_report,
)
from src.experiments.final_publication_scenarios.statistics import edge_sensitivity_stats
from src.experiments.final_publication_scenarios.tables import generate_all_tables
from src.experiments.result_writer import load_experiment_config
from src.utils.paths import resolve_project_root

CONFIG_SRC = Path("new_experiments/final_validated_runs/configs/final_validated_runs.yaml")


def _write_campaign_size_validation(out: Path, comp: pd.DataFrame, run_level: pd.DataFrame) -> None:
    fcgnn = run_level[run_level["method"] == "fcgnn"]
    lines = [
        "# Campaign-size control validation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    checks = []
    if not comp.empty:
        checks.append(("All runs have 200 descriptors", (comp["descriptor_count"] == 200).all()))
    if "graph_nodes" in fcgnn.columns:
        checks.append(("Graph nodes = 200", (fcgnn["graph_nodes"] == 200).all()))
    seeds = sorted(fcgnn["seed"].unique().tolist()) if not fcgnn.empty else []
    checks.append(("Ten seeds present (fcgnn)", set(REQUIRED_SEEDS).issubset(set(seeds))))
    for name, ok in checks:
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.append(f"\n**Overall:** {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
    (out / "validation/campaign_size_control_validation.md").write_text("\n".join(lines), encoding="utf-8")


def _completeness_report(out: Path, tables: list[str], figures: list[str]) -> None:
    required_tables = [f"table_T{i}" for i in range(1, 11)]
    required_figures = [f"figure_F{i}" for i in range(1, 7)]
    rows = []
    for t in required_tables:
        found = any(t in g for g in tables)
        rows.append({"artifact": t, "status": "present" if found else "missing", "validated": found, "placement": "main paper"})
    for f in required_figures:
        found = any(f in g for g in figures)
        rows.append({"artifact": f, "status": "present" if found else "missing", "validated": found, "placement": "main paper"})
    df = pd.DataFrame(rows)
    lines = [
        "# Publication artifact completeness",
        "",
        f"S0–S4 tables present: {all(f'table_T{i}' in str(tables) for i in range(1, 5))}",
        f"Campaign-size tables: {any('T5' in t or 'T6' in t or 'T7' in t for t in tables)}",
        f"Edge-sensitivity tables: {any('T8' in t or 'T9' in t for t in tables)}",
        f"Statistical table: {'table_T10' in str(tables)}",
        f"Figures generated: {len(figures)}",
        "",
        df.to_markdown(index=False),
    ]
    (out / "validation/publication_artifact_completeness.md").write_text("\n".join(lines), encoding="utf-8")


def run_publication_package(*, skip_edge: bool = False, edge_dry_only: bool = False) -> dict:
    project_root = resolve_project_root()
    guard = PublicationScenariosGuard(project_root)
    guard.ensure_directory_tree()
    out = guard.output_root

    inventory = build_source_inventory(project_root)
    inventory.to_csv(out / "audit/source_results_inventory.csv", index=False)
    write_source_selection_report(out / "audit/source_selection_report.md", inventory)

    if CONFIG_SRC.exists():
        shutil.copy(project_root / CONFIG_SRC, out / "configs/final_publication_scenarios.yaml")
    config = load_experiment_config(out / "configs/final_publication_scenarios.yaml") if (out / "configs/final_publication_scenarios.yaml").exists() else load_experiment_config(CONFIG_SRC)

    scenario = collect_scenario_results(project_root)
    scen_dir = out / "results/scenarios"
    for name, df in scenario.items():
        if not df.empty:
            df.to_csv(scen_dir / f"{name}.csv", index=False)

    campaign = collect_campaign_size_results(project_root)
    cs_dir = out / "results/campaign_size"
    for name, df in campaign.items():
        if not df.empty:
            df.to_csv(cs_dir / f"{name}.csv", index=False)
    _write_campaign_size_validation(out, campaign.get("composition_validation", pd.DataFrame()), campaign.get("run_level_metrics", pd.DataFrame()))

    edge_df = pd.DataFrame()
    if not skip_edge:
        edge_df = run_edge_sensitivity(project_root, out, config, dry_only=edge_dry_only)
        if not edge_df.empty:
            edge_df.to_csv(out / "results/edge_sensitivity/run_level_metrics.csv", index=False)
            edge_df.groupby(["scenario", "unique_edges"]).mean(numeric_only=True).reset_index().to_csv(
                out / "results/edge_sensitivity/summary_mean_std.csv", index=False
            )
            edge_stats = edge_sensitivity_stats(edge_df)
            edge_stats.to_csv(out / "results/edge_sensitivity/statistical_tests.csv", index=False)

    all_stats = pd.concat([
        scenario.get("statistical_tests", pd.DataFrame()),
        campaign.get("statistical_tests", pd.DataFrame()),
        edge_sensitivity_stats(edge_df) if not edge_df.empty else pd.DataFrame(),
    ], ignore_index=True)
    if not all_stats.empty:
        all_stats.to_csv(out / "statistical_analysis/primary_statistical_tests.csv", index=False)

    fcgnn_cs = campaign.get("fcgnn_run_level_metrics", pd.DataFrame())
    campaign_cost = fcgnn_cs.groupby("campaign_size").agg(
        Nodes=("graph_nodes", "mean"),
        graph_build_time=("runtime_graph_construction_sec", "mean"),
        inference_time=("runtime_gnn_inference_sec", "mean"),
    ).reset_index() if not fcgnn_cs.empty else pd.DataFrame()

    edge_perf = edge_df.groupby(["scenario", "unique_edges"]).agg(
        campaign_precision=("campaign_precision", "mean"),
        campaign_recall=("campaign_recall", "mean"),
        campaign_f1=("campaign_f1", "mean"),
        false_campaign_rate=("false_campaign_alert_rate", "mean"),
    ).reset_index() if not edge_df.empty else pd.DataFrame()

    edge_cost = edge_df.groupby("unique_edges").agg(
        graph_build_time=("graph_build_time", "mean"),
        inference_time=("inference_time", "mean"),
    ).reset_index() if not edge_df.empty else pd.DataFrame()

    tables = generate_all_tables(
        out,
        safety=scenario.get("scenario_safety_metrics", pd.DataFrame()),
        fleet=scenario.get("fleet_campaign_metrics", pd.DataFrame()),
        weak=scenario.get("weak_campaign_support", pd.DataFrame()),
        campaign_strong=campaign.get("strong_summary", pd.DataFrame()),
        campaign_weak=campaign.get("weak_summary", pd.DataFrame()),
        campaign_cost=campaign_cost,
        edge_perf=edge_perf,
        edge_cost=edge_cost,
        stats=all_stats,
    )
    figures = generate_all_figures(
        out / "figures",
        weak=scenario.get("weak_campaign_support", pd.DataFrame()),
        campaign_fcgnn=fcgnn_cs,
        edge_df=edge_df,
    )
    _completeness_report(out, tables, figures)

    summary = _write_summary(out, scenario, campaign, edge_df, inventory)
    return summary


def _write_summary(out, scenario, campaign, edge_df, inventory) -> dict:
    safety = scenario.get("scenario_safety_metrics", pd.DataFrame())
    fleet = scenario.get("fleet_campaign_metrics", pd.DataFrame())
    fcgnn = campaign.get("fcgnn_run_level_metrics", pd.DataFrame())

    def _mean(df, col, filt=None):
        if df.empty or col not in df.columns:
            return None
        sub = df if filt is None else filt(df)
        return float(sub[col].mean())

    info = {
        "s0_false_campaign": _mean(safety, "false_campaign_alert_rate", lambda d: d[d["scenario_id"] == "S0"]),
        "s3_campaign_f1": _mean(fleet, "campaign_f1", lambda d: d[d["scenario_id"] == "S3"]),
        "s4_campaign_f1": _mean(fleet, "campaign_f1", lambda d: d[d["scenario_id"] == "S4"]),
        "campaign_size_runs": len(fcgnn),
        "edge_runs": len(edge_df),
        "eligible_sources": int(inventory["eligible_for_final_publication"].sum()),
    }
    lines = [
        "# Final scenario experiment summary",
        "",
        "## Safety (C3 GraphSAGE)",
        f"1. S0 false campaign alerts: mean rate = {info['s0_false_campaign']}",
        f"2. S1 isolated: see scenario_safety_metrics.csv",
        f"3. S2 merging: see scenario_safety_metrics.csv",
        f"4. S3 strong campaign F1: {info['s3_campaign_f1']}",
        f"5. S4 weak campaign F1: {info['s4_campaign_f1']}",
        "",
        "## Campaign size (fcgnn, corrected)",
        f"6–7. See results/campaign_size/",
        "",
        "## Edge sensitivity",
        f"8–11. Edge runs completed: {info['edge_runs']}",
        "",
        "## Publication readiness",
        "See validation/publication_artifact_completeness.md",
    ]
    (out / "FINAL_SCENARIO_EXPERIMENT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return info
