"""Execute a single scenario × method × seed run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.method_descriptor_clustering import run_descriptor_clustering_method
from src.experiments.method_fcgnn import run_fcgnn_method
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.method_standard_gnn import run_standard_gnn_method
from src.experiments.result_writer import ExperimentRunContext
from src.experiments.scenario_generator import generate_scenario_records
from src.experiments.scenario_registry import ScenarioSpec, get_scenario


def run_single_experiment(
    ctx: ExperimentRunContext,
    spec: ScenarioSpec,
    method_id: str,
    config: dict[str, Any],
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    similarity_threshold: float | None = None,
    max_neighbors: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scenario_df, membership = generate_scenario_records(
        spec,
        seed=ctx.seed,
        campaign_size=ctx.campaign_size or 0,
        coordination_strength=ctx.coordination_strength or 0.0,
        descriptors=descriptors,
        manifest=manifest,
        config=config,
    )

    if method_id == "local_ids":
        outputs = run_local_ids_method(ctx, scenario_df, membership, config)
    elif method_id == "descriptor_clustering":
        outputs = run_descriptor_clustering_method(
            ctx, scenario_df, membership, config,
            seed=ctx.seed,
            similarity_threshold=similarity_threshold,
            max_neighbors=max_neighbors,
        )
    elif method_id == "standard_gnn":
        outputs = run_standard_gnn_method(
            ctx, scenario_df, membership, config,
            seed=ctx.seed,
            similarity_threshold=similarity_threshold,
            max_neighbors=max_neighbors,
        )
    elif method_id == "fcgnn":
        outputs = run_fcgnn_method(
            ctx, scenario_df, membership, config,
            seed=ctx.seed,
            similarity_threshold=similarity_threshold,
            max_neighbors=max_neighbors,
        )
    else:
        raise ValueError(f"Unknown method: {method_id}")

    metrics = aggregate_run_metrics(
        method=method_id,
        seed=ctx.seed,
        scenario_key=spec.key,
        campaign_size=ctx.campaign_size or 0,
        coordination_strength=ctx.coordination_strength or 0.0,
        event_predictions=outputs.event_predictions,
        vehicle_predictions=outputs.vehicle_predictions,
        membership=membership,
        cluster_df=outputs.cluster_df,
        expect_campaign=spec.expect_coordinated_campaign,
        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
    )

    _write_run_outputs(ctx, spec, scenario_df, membership, outputs, metrics)
    return metrics


def _write_run_outputs(ctx, spec, scenario_df, membership, outputs, metrics) -> None:
    rd = ctx.run_dir
    scenario_df.to_csv(rd / "selected_source_records.csv", index=False)
    membership.to_csv(rd / "scenario_membership.csv", index=False)
    outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
    outputs.vehicle_predictions.to_csv(rd / "vehicle_predictions.csv", index=False)
    outputs.campaign_predictions.to_csv(rd / "campaign_predictions.csv", index=False)
    if not outputs.graph_stats.empty:
        outputs.graph_stats.to_csv(rd / "graph_statistics.csv", index=False)
    if not outputs.edge_list.empty:
        outputs.edge_list.to_csv(rd / "edge_list.csv", index=False)
    pd.DataFrame([metrics]).to_csv(rd / "run_level_metrics.csv", index=False)
    (rd / "runtime_memory.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    if outputs.embeddings is not None:
        emb_dir = ctx.output_root / "embeddings" / spec.key / ctx.run_id
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_df = pd.DataFrame(outputs.embeddings)
        emb_df.insert(0, "event_id", outputs.event_predictions["event_id"].values[: len(emb_df)])
        emb_df.to_csv(emb_dir / "node_embeddings.csv", index=False)
