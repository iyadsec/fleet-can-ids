#!/usr/bin/env python3
"""Fast scenario-only regeneration from existing set_pilot artifacts."""

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


def load_features(work: Path) -> pd.DataFrame:
    parquet = work / "windows" / "all_window_features.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    shard_dir = work / "windows" / "feature_shards"
    shards = sorted(shard_dir.glob("*.parquet"))
    if shards:
        return pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    raise FileNotFoundError(f"No feature artifacts under {work / 'windows'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", default="set_01")
    args = parser.parse_args()

    work = set_work_root(OUTPUT_ROOT, args.set_id)
    set_id = args.set_id

    tracemalloc.start()
    t0 = time.perf_counter()

    features = load_features(work)
    pred_df = pd.read_csv(work / "results" / "local_detection" / "window_predictions.csv")
    desc_path = work / "descriptors" / f"{set_id}_fleet_candidate_descriptors.csv"
    if not desc_path.exists():
        desc_path = work / "descriptors" / "fleet_candidate_descriptors.csv"
    desc_df = pd.read_csv(desc_path)
    window_manifest = pd.read_csv(work / "manifests" / "window_manifest.csv")
    nm = pd.read_csv(work / "manifests" / "normalization_manifest.csv")
    gs = pd.read_csv(work / "graph" / f"{set_id}_graph_statistics.csv").iloc[0].to_dict()
    desc_summary = pd.read_csv(work / "results" / "descriptor_transfer" / "communication_summary.csv")
    inv = pd.read_csv(OUTPUT_ROOT / "manifests" / "ctt_file_inventory.csv")
    caps = json.loads((work / "manifests" / f"stage_set_pilot_{set_id}_complete.json").read_text()).get(
        "caps",
        {"max_rows_per_file": 475_000, "max_windows": 800_000, "max_descriptors": 100_000},
    )

    scenario_results = run_scenario_evaluation(
        features, pred_df, desc_df, work,
        scenarios=SET_PILOT_SCENARIOS, seeds=SCENARIO_SEEDS, target_set=set_id,
    )
    _rename_scenario_outputs(work / "results" / "scenario_evaluation", set_id, SET_PILOT_SCENARIOS)

    campaign_size = run_set_campaign_size_sensitivity(features, pred_df, set_id, work)
    edge_sensitivity = run_set_edge_sensitivity(desc_df, set_id, work)
    run_statistical_analysis(scenario_results, campaign_size, work / "statistics")

    table_names = generate_set_tables(
        set_id, inv, window_manifest, pd.read_csv(work / "results" / "local_detection" / f"{set_id}_by_subset.csv"),
        desc_summary, gs, scenario_results, campaign_size, edge_sensitivity, work, caps,
    )
    figure_names = generate_set_figures(
        set_id, pred_df,
        pd.read_csv(work / "results" / "local_detection" / f"{set_id}_by_subset.csv"),
        gs, scenario_results, campaign_size, edge_sensitivity, work,
    )

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    generate_set_summary(
        set_id, work, {}, caps, nm, window_manifest,
        pd.read_csv(work / "results" / "local_detection" / f"{set_id}_by_subset.csv"),
        desc_summary, gs, scenario_results, campaign_size, edge_sensitivity,
        elapsed, peak / (1024 * 1024), table_names, figure_names,
    )

    marker = work / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    if marker.exists():
        payload = json.loads(marker.read_text())
        payload["scenario_metrics_version"] = "v3_local_fleet_split"
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Scenario means:")
    print(
        scenario_results.groupby("scenario")[
            ["local_or_incident_detected", "fleet_campaign_detected", "false_campaign", "campaign_f1"]
        ].mean().round(4)
    )
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
