"""Staged execution: audit, pilot, and full pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ctt.audit import run_dataset_audit
from src.ctt.constants import OUTPUT_ROOT
from src.ctt.descriptors import generate_descriptors
from src.ctt.evaluation import (
    generate_figures,
    generate_publication_tables,
    run_campaign_size_sensitivity,
    run_edge_sensitivity,
)
from src.ctt.features import write_feature_schema
from src.ctt.fleet_campaign import write_fleet_transfer_policy
from src.ctt.fleet_graph import build_behavioural_graph, save_graph_artifacts
from src.ctt.local_detector import run_local_onboarding
from src.ctt.progress_logger import ProgressLogger
from src.ctt.run_config import RunConfig
from src.ctt.scenarios import run_scenario_evaluation
from src.ctt.splits import run_split_validation
from src.ctt.statistics import run_statistical_analysis
from src.ctt.streaming_pipeline import run_streaming_pipeline
from src.ctt.utils import ensure_dir, write_markdown


def _write_stage_marker(cfg: RunConfig, extra: dict | None = None) -> None:
    marker = cfg.stage_marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": cfg.stage, "status": "complete", **(extra or {})}
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _check_prior_stages(cfg: RunConfig) -> tuple[bool, str]:
    """Full stage requires audit and pilot completion markers."""
    audit_marker = cfg.output_root / "manifests" / "stage_audit_complete.json"
    pilot_marker = cfg.output_root / "manifests" / "stage_pilot_complete.json"
    if cfg.stage != "full":
        return True, ""
    missing = []
    if not audit_marker.exists():
        missing.append("audit")
    if not pilot_marker.exists():
        missing.append("pilot")
    if missing:
        return False, f"Stage 3 blocked: complete Stage 1 and Stage 2 first (missing: {', '.join(missing)})"
    return True, ""


def run_stage_audit(cfg: RunConfig, progress: ProgressLogger) -> dict:
    """Stage 1: dataset audit only — no normalize, window, train, or graph."""
    progress.info("Stage 1: dataset inventory and schema audit (no normalization)")
    audit_result = run_dataset_audit(cfg.dataset_root, cfg.output_root)
    if not audit_result["suitability"]["schema_normalizable"]:
        raise RuntimeError("Schema cannot be normalized; aborting.")
    progress.info("Split policy manifest (metadata only, no row processing)")
    run_split_validation(cfg.dataset_root, cfg.output_root)
    write_markdown(
        cfg.output_root / "audit" / "stage_execution_policy.md",
        "Staged Execution Policy",
        {
            "Stage 1 (audit)": "Inventory, vehicles, attacks, schema, fast row counts. No normalization.",
            "Stage 2 (pilot)": (
                f"set={cfg.pilot_dataset_set}, train={cfg.pilot_train_subset}, "
                f"test={cfg.pilot_test_subset}, max_rows={cfg.max_rows_per_file}, "
                f"max_windows={cfg.max_windows}, max_descriptors={cfg.max_descriptors}"
            ),
            "Stage 3 (full)": "Blocked until Stage 1 and Stage 2 validation pass.",
        },
    )
    _write_stage_marker(cfg, {
        "total_files": audit_result["total_files"],
        "total_rows": audit_result["total_rows"],
        "vehicles": audit_result["vehicles_found"],
        "attacks": audit_result["attacks_found"],
    })
    return audit_result


def run_stage_pilot(cfg: RunConfig, progress: ProgressLogger) -> dict:
    """Stage 2: capped pilot run on one train + one test subset."""
    progress.info(
        f"Stage 2 pilot scope: {cfg.pilot_dataset_set}/{cfg.pilot_train_subset} + "
        f"{cfg.pilot_test_subset}"
    )
    write_feature_schema(cfg.output_root)
    _, window_manifest, features = run_streaming_pipeline(
        cfg.dataset_root, cfg.output_root, config=cfg, progress=progress
    )
    progress.info("Local benign-only onboarding and detection")
    metrics_df, pred_df, _ = run_local_onboarding(window_manifest, cfg.output_root, features=features)

    progress.info("Descriptor generation (capped)")
    desc_df = generate_descriptors(
        pred_df, features, cfg.output_root, max_descriptors=cfg.max_descriptors
    )

    progress.info("Fleet graph (behavioural similarity only)")
    write_fleet_transfer_policy(cfg.output_root)
    if len(desc_df) > (cfg.max_graph_nodes or len(desc_df)):
        desc_df = desc_df.head(cfg.max_graph_nodes)
    node_df, edge_df, graph_stats = build_behavioural_graph(desc_df)
    save_graph_artifacts(node_df, edge_df, graph_stats, cfg.output_root)

    pilot_scenarios = ["benign_fleet_control", "isolated_attack", "strong_campaign"]
    progress.info(f"Pilot scenarios: {pilot_scenarios}")
    scenario_results = run_scenario_evaluation(
        features, pred_df, desc_df, cfg.output_root,
        scenarios=pilot_scenarios, seeds=[11], progress=progress,
    )

    _write_stage_marker(cfg, {
        "windows": len(window_manifest),
        "descriptors": len(desc_df),
        "scenarios": pilot_scenarios,
    })
    return {
        "window_manifest": window_manifest,
        "features": features,
        "metrics_df": metrics_df,
        "pred_df": pred_df,
        "desc_df": desc_df,
        "graph_stats": graph_stats,
        "scenario_results": scenario_results,
    }


def run_stage_full(cfg: RunConfig, progress: ProgressLogger, audit_result: dict | None = None) -> dict:
    """Stage 3: full cross-dataset validation."""
    ok, msg = _check_prior_stages(cfg)
    if not ok:
        raise RuntimeError(msg)

    if audit_result is None:
        audit_result = run_dataset_audit(cfg.dataset_root, cfg.output_root)

    run_split_validation(cfg.dataset_root, cfg.output_root)
    write_feature_schema(cfg.output_root)
    _, window_manifest, features = run_streaming_pipeline(
        cfg.dataset_root, cfg.output_root, config=cfg, progress=progress
    )
    metrics_df, pred_df, _ = run_local_onboarding(window_manifest, cfg.output_root, features=features)
    desc_df = generate_descriptors(pred_df, features, cfg.output_root, max_descriptors=cfg.max_descriptors)
    write_fleet_transfer_policy(cfg.output_root)
    if cfg.max_graph_nodes and len(desc_df) > cfg.max_graph_nodes:
        desc_df = desc_df.head(cfg.max_graph_nodes)
    node_df, edge_df, graph_stats = build_behavioural_graph(desc_df)
    save_graph_artifacts(node_df, edge_df, graph_stats, cfg.output_root)
    scenario_results = run_scenario_evaluation(
        features, pred_df, desc_df, cfg.output_root, progress=progress
    )
    campaign_size = run_campaign_size_sensitivity(features, pred_df, cfg.output_root)
    edge_sensitivity = run_edge_sensitivity(desc_df, cfg.output_root)
    run_statistical_analysis(scenario_results, campaign_size, cfg.output_root)
    desc_summary = pd.read_csv(cfg.output_root / "results" / "descriptor_transfer" / "communication_summary.csv")
    generate_publication_tables(
        audit_result["file_inventory"], window_manifest, metrics_df, desc_summary,
        graph_stats, scenario_results, campaign_size, edge_sensitivity, cfg.output_root,
    )
    generate_figures(
        window_manifest, pred_df, metrics_df, graph_stats,
        scenario_results, campaign_size, edge_sensitivity, desc_summary, cfg.output_root,
    )
    _write_stage_marker(cfg, {"windows": len(window_manifest), "descriptors": len(desc_df)})
    return {
        "audit_result": audit_result,
        "window_manifest": window_manifest,
        "metrics_df": metrics_df,
        "desc_df": desc_df,
        "graph_stats": graph_stats,
        "scenario_results": scenario_results,
        "campaign_size": campaign_size,
        "edge_sensitivity": edge_sensitivity,
    }
