#!/usr/bin/env python3
"""Resume set_pilot from graph/scenarios onward using existing features and descriptors."""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OUTPUT_ROOT, SCENARIO_SEEDS
from src.ctt.local_detector import run_local_onboarding
from src.ctt.run_config import RunConfig
from src.ctt.set_pilot import (
    SET_PILOT_SCENARIOS,
    _rename_scenario_outputs,
    generate_set_figures,
    generate_set_summary,
    generate_set_tables,
    run_set_campaign_size_sensitivity,
    run_set_edge_sensitivity,
    save_set_descriptors,
    save_set_graph,
    save_set_local_detection,
    set_work_root,
)
from src.ctt.descriptors import generate_descriptors
from src.ctt.fleet_campaign import write_fleet_transfer_policy
from src.ctt.fleet_graph import build_behavioural_graph
from src.ctt.scenarios import run_scenario_evaluation
from src.ctt.statistics import run_statistical_analysis
from src.ctt.streaming_pipeline import run_streaming_pipeline


def main() -> int:
    set_id = "set_01"
    cfg = RunConfig.for_stage(
        "set_pilot",
        set_id=set_id,
        max_rows_per_file=475_000,
        max_windows=800_000,
        max_descriptors=100_000,
        resume=True,
        skip_existing=True,
    )
    work = set_work_root(cfg.output_root, set_id)

    tracemalloc.start()
    t0 = time.perf_counter()

    _, window_manifest, features = run_streaming_pipeline(
        cfg.dataset_root, work, config=cfg, progress=None
    )
    metrics_df, pred_df, _ = run_local_onboarding(window_manifest, work, features=features)
    save_set_local_detection(metrics_df, set_id, work)

    desc_df = generate_descriptors(pred_df, features, work, max_descriptors=cfg.max_descriptors)
    meta_df = pd.read_csv(work / "descriptors" / "descriptor_metadata.csv")
    save_set_descriptors(desc_df, meta_df, set_id, work)

    balance = desc_df.groupby(["vehicle_id", "subset_name", "attack_type"]).size().reset_index(name="count")
    balance.to_csv(work / "audit" / f"{set_id}_descriptor_balance.csv", index=False)

    write_fleet_transfer_policy(work)
    node_df, edge_df, graph_stats = build_behavioural_graph(desc_df, cross_vehicle_cap=20)
    save_set_graph(node_df, edge_df, graph_stats, set_id, work)

    scenario_results = run_scenario_evaluation(
        features, pred_df, desc_df, work,
        scenarios=SET_PILOT_SCENARIOS, seeds=SCENARIO_SEEDS, target_set=set_id,
    )
    _rename_scenario_outputs(work / "results" / "scenario_evaluation", set_id, SET_PILOT_SCENARIOS)

    campaign_size = run_set_campaign_size_sensitivity(features, pred_df, set_id, work)
    edge_sensitivity = run_set_edge_sensitivity(desc_df, set_id, work)
    run_statistical_analysis(scenario_results, campaign_size, work / "statistics")

    desc_summary = pd.read_csv(work / "results" / "descriptor_transfer" / "communication_summary.csv")
    inv = pd.read_csv(cfg.output_root / "manifests" / "ctt_file_inventory.csv")
    caps = {"max_rows_per_file": 475_000, "max_windows": 800_000, "max_descriptors": 100_000}
    nm = pd.read_csv(work / "manifests" / "normalization_manifest.csv")

    table_names = generate_set_tables(
        set_id, inv, window_manifest, metrics_df, desc_summary,
        graph_stats, scenario_results, campaign_size, edge_sensitivity, work, caps,
    )
    figure_names = generate_set_figures(
        set_id, pred_df, metrics_df, graph_stats, scenario_results, campaign_size, edge_sensitivity, work,
    )

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    generate_set_summary(
        set_id, work, {}, caps, nm, window_manifest, metrics_df, desc_summary,
        graph_stats, scenario_results, campaign_size, edge_sensitivity,
        elapsed, peak / (1024 * 1024), table_names, figure_names,
    )

    marker = work / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    marker.write_text(
        json.dumps(
            {
                "stage": "set_pilot", "set_id": set_id, "status": "complete",
                "files_processed": len(nm), "windows": len(window_manifest),
                "descriptors": len(desc_df), "caps": caps,
                "runtime_sec": elapsed, "peak_memory_mb": peak / (1024 * 1024),
                "cross_vehicle_edge_pct": graph_stats.get("cross_vehicle_edge_pct"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done in {elapsed:.1f}s peak={peak/1e6:.1f}MB cross_pct={graph_stats.get('cross_vehicle_edge_pct')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
