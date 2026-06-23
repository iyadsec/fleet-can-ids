#!/usr/bin/env python3
"""Complete set_pilot summary/marker after partial run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.run_config import RunConfig
from src.ctt.set_pilot import (
    SET_PILOT_SCENARIOS,
    estimate_set_scope,
    generate_set_figures,
    generate_set_summary,
    generate_set_tables,
    set_work_root,
)


def main() -> int:
    set_id = "set_01"
    cfg = RunConfig.for_stage("set_pilot", set_id=set_id)
    work = set_work_root(cfg.output_root, set_id)
    scope = estimate_set_scope(cfg.dataset_root, set_id)

    caps_path = work / "audit" / "set_pilot_scope.md"
    caps = {"max_rows_per_file": 1_000_000, "max_windows": 400_000, "max_descriptors": 50_000}

    norm_manifest = pd.read_csv(work / "manifests" / "normalization_manifest.csv")
    window_manifest = pd.read_csv(work / "manifests" / "window_manifest.csv")
    metrics_df = pd.read_csv(work / "results" / "local_detection" / "by_set_and_subset.csv")
    pred_df = pd.read_csv(work / "results" / "local_detection" / "window_predictions.csv")
    desc_summary = pd.read_csv(work / "results" / "descriptor_transfer" / "communication_summary.csv")
    graph_stats = pd.read_csv(work / "graph" / f"{set_id}_graph_statistics.csv").iloc[0].to_dict()
    scenario_results = pd.read_csv(work / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv")
    campaign_size = pd.read_csv(work / "results" / "campaign_size" / f"{set_id}_run_level.csv")
    edge_sensitivity = pd.read_csv(work / "results" / "edge_sensitivity" / f"{set_id}_run_level.csv")
    inv = pd.read_csv(cfg.output_root / "manifests" / "ctt_file_inventory.csv")

    table_names = generate_set_tables(
        set_id, inv, window_manifest, metrics_df, desc_summary,
        graph_stats, scenario_results, campaign_size, edge_sensitivity, work, caps,
    )
    figure_names = generate_set_figures(
        set_id, pred_df, metrics_df, graph_stats, scenario_results, campaign_size, edge_sensitivity, work,
    )
    generate_set_summary(
        set_id, work, scope, caps, norm_manifest, window_manifest, metrics_df,
        desc_summary, graph_stats, scenario_results, campaign_size, edge_sensitivity,
        0.0, 0.0, table_names, figure_names,
    )

    marker = work / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    marker.write_text(
        json.dumps(
            {
                "stage": "set_pilot",
                "set_id": set_id,
                "status": "complete",
                "files_processed": len(norm_manifest),
                "windows": len(window_manifest),
                "descriptors": len(pd.read_csv(work / "descriptors" / f"{set_id}_fleet_candidate_descriptors.csv")),
                "caps": caps,
                "scenarios": SET_PILOT_SCENARIOS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Completed summary and marker at {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
