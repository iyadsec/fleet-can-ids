#!/usr/bin/env python3
"""Regenerate scenario metrics and downstream artifacts after metric schema changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OUTPUT_ROOT, SCENARIO_SEEDS
from src.ctt.run_config import RunConfig
from src.ctt.set_pilot import (
    SET_PILOT_SCENARIOS,
    _rename_scenario_outputs,
    generate_set_figures,
    generate_set_summary,
    generate_set_tables,
    run_set_campaign_size_sensitivity,
    run_set_edge_sensitivity,
    set_work_root,
)
from src.ctt.scenarios import run_scenario_evaluation
from src.ctt.statistics import run_statistical_analysis
from src.ctt.streaming_pipeline import run_streaming_pipeline
from src.ctt.local_detector import run_local_onboarding
from src.ctt.descriptors import generate_descriptors


def regenerate_set_pilot_scenarios(
    set_id: str = "set_01",
    max_rows_per_file: int = 475_000,
    max_windows: int = 800_000,
    max_descriptors: int = 100_000,
) -> int:
    cfg = RunConfig.for_stage(
        "set_pilot",
        set_id=set_id,
        max_rows_per_file=max_rows_per_file,
        max_windows=max_windows,
        max_descriptors=max_descriptors,
        resume=True,
        skip_existing=True,
        confirm_large_run=True,
    )
    work = set_work_root(cfg.output_root, set_id)

    tracemalloc.start()
    t0 = time.perf_counter()

    _, window_manifest, features = run_streaming_pipeline(
        cfg.dataset_root, work, config=cfg, progress=None
    )
    metrics_df, pred_df, _ = run_local_onboarding(window_manifest, work, features=features)
    desc_df = generate_descriptors(pred_df, features, work, max_descriptors=max_descriptors)

    scenario_results = run_scenario_evaluation(
        features,
        pred_df,
        desc_df,
        work,
        scenarios=SET_PILOT_SCENARIOS,
        seeds=SCENARIO_SEEDS,
        target_set=set_id,
    )
    _rename_scenario_outputs(work / "results" / "scenario_evaluation", set_id, SET_PILOT_SCENARIOS)

    campaign_size = run_set_campaign_size_sensitivity(features, pred_df, set_id, work)
    edge_sensitivity = run_set_edge_sensitivity(desc_df, set_id, work)
    run_statistical_analysis(scenario_results, campaign_size, work / "statistics")

    desc_summary = pd.read_csv(work / "results" / "descriptor_transfer" / "communication_summary.csv")
    inv = pd.read_csv(cfg.output_root / "manifests" / "ctt_file_inventory.csv")
    caps = {
        "max_rows_per_file": max_rows_per_file,
        "max_windows": max_windows,
        "max_descriptors": max_descriptors,
    }
    nm = pd.read_csv(work / "manifests" / "normalization_manifest.csv")
    gs = pd.read_csv(work / "graph" / f"{set_id}_graph_statistics.csv").iloc[0].to_dict()

    table_names = generate_set_tables(
        set_id, inv, window_manifest, metrics_df, desc_summary,
        gs, scenario_results, campaign_size, edge_sensitivity, work, caps,
    )
    figure_names = generate_set_figures(
        set_id, pred_df, metrics_df, gs, scenario_results, campaign_size, edge_sensitivity, work,
    )

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    generate_set_summary(
        set_id, work, {}, caps, nm, window_manifest, metrics_df, desc_summary,
        gs, scenario_results, campaign_size, edge_sensitivity,
        elapsed, peak / (1024 * 1024), table_names, figure_names,
    )

    marker = work / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        payload["scenario_metrics_version"] = "v3_local_fleet_split"
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Regenerated scenario metrics for {set_id} in {elapsed:.1f}s")
    print(scenario_results.groupby("scenario")[
        ["local_or_incident_detected", "fleet_campaign_detected", "false_campaign", "campaign_f1"]
    ].mean().round(4))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", default="set_01")
    parser.add_argument("--max-rows-per-file", type=int, default=475_000)
    parser.add_argument("--max-windows", type=int, default=800_000)
    parser.add_argument("--max-descriptors", type=int, default=100_000)
    args = parser.parse_args()
    return regenerate_set_pilot_scenarios(
        args.set_id, args.max_rows_per_file, args.max_windows, args.max_descriptors
    )


if __name__ == "__main__":
    raise SystemExit(main())
