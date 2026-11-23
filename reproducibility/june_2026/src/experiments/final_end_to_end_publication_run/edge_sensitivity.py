"""Edge-connectivity sensitivity using fixed scenario records from this run."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.coordinated_campaign_refinement.refinement_pipeline import run_refinement_fcgnn
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration
from src.experiments.final_end_to_end_publication_run.scenario_registry import REQUIRED_SEEDS
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown

THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
MAX_CROSS = [3, 5, 10, 20]


def run_edge_sensitivity(
    *,
    scenario_records: dict[str, pd.DataFrame],
    memberships: dict[str, pd.DataFrame],
    config: dict[str, Any],
    shared: SharedFleetConfiguration,
    output_root: Path,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    run_root = output_root / "results" / "edge_sensitivity" / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    seeds = seeds or list(REQUIRED_SEEDS)
    rows: list[dict] = []
    k_same = shared.max_same_vehicle_neighbors

    for test_condition, scenario_df in scenario_records.items():
        membership = memberships[test_condition]
        for seed in seeds:
            for tau in THRESHOLDS:
                for k_cross in MAX_CROSS:
                    cfg = SharedFleetConfiguration(
                        **{
                            **shared.to_dict(),
                            "similarity_threshold": tau,
                            "max_cross_vehicle_neighbors": k_cross,
                            "max_same_vehicle_neighbors": k_same,
                        }
                    )
                    run_id = f"edge_{test_condition.replace(' ', '_')}_seed{seed}_tau{tau:.2f}_k{k_cross}"
                    t0 = time.perf_counter()
                    try:
                        ensure_fleet_scaler_in_config(config, scenario_df, membership)
                        outputs, _, _, _ = run_refinement_fcgnn(
                            scenario_df, membership, config, seed, cfg.to_refinement_design()
                        )
                        gs = outputs.graph_stats.iloc[0].to_dict() if not outputs.graph_stats.empty else {}
                        br = compute_false_campaign_breakdown(
                            outputs.event_predictions, membership, outputs.cluster_df, expect_campaign=True
                        )
                        rows.append(
                            {
                                "run_id": run_id,
                                "test_condition": test_condition,
                                "seed": seed,
                                "similarity_threshold": tau,
                                "max_cross_vehicle_neighbors": k_cross,
                                "max_same_vehicle_neighbors": k_same,
                                "unique_edges": gs.get("unique_undirected_edges", gs.get("graph_unique_undirected_edges")),
                                "campaign_precision": br.get("campaign_precision"),
                                "campaign_recall": br.get("campaign_recall"),
                                "campaign_f1": br.get("campaign_f1"),
                                "membership_precision": br.get("campaign_membership_precision"),
                                "fragmentation": br.get("fragmentation"),
                                "incorrect_merging": br.get("incorrect_merging"),
                                "false_campaign_rate": float(br.get("false_campaign_cluster_count", 0) > 0),
                                "graph_build_time": outputs.runtime.get("graph_construction_sec"),
                                "gnn_inference_time": outputs.runtime.get("gnn_inference_sec"),
                                "end_to_end_latency": time.perf_counter() - t0,
                                "peak_memory": outputs.runtime.get("peak_memory_mb"),
                            }
                        )
                    except Exception:
                        (output_root / "logs" / f"{run_id}.log").write_text(traceback.format_exc())
    df = pd.DataFrame(rows)
    if not df.empty:
        out = output_root / "results" / "edge_sensitivity"
        df.to_csv(out / "run_level_metrics.csv", index=False)
        df.groupby("unique_edges").mean(numeric_only=True).reset_index().to_csv(out / "graph_statistics.csv", index=False)
        df.groupby(["test_condition", "unique_edges"]).mean(numeric_only=True).reset_index().to_csv(out / "campaign_metrics.csv", index=False)
        df[["graph_build_time", "gnn_inference_time", "end_to_end_latency", "peak_memory"]].describe().to_csv(out / "runtime_memory.csv")
    return df
