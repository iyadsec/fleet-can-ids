"""Execute single campaign-analysis experiment runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    build_event_budget_validation_row,
    build_vehicle_composition_row,
    generate_corrected_campaign_scenario,
)
from src.experiments.campaign_analysis_generator import generate_campaign_analysis_scenario
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config, resolve_fleet_scaler_from_config
from src.experiments.campaign_analysis_writer import CampaignRunContext
from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.experiment_pipeline import resolve_vehicle_id_column
from src.experiments.method_descriptor_clustering import run_descriptor_clustering_method
from src.experiments.method_fcgnn import run_fcgnn_method
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.method_standard_gnn import run_standard_gnn_method
from src.experiments.result_writer import ExperimentRunContext
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


def _adapter_ctx(ctx: CampaignRunContext) -> ExperimentRunContext:
    """Adapt campaign context for existing method runners."""
    return ExperimentRunContext(
        run_id=ctx.run_id,
        scenario_key=f"{ctx.experiment}_{ctx.attack_strength}",
        method=ctx.method,
        seed=ctx.seed,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        created_at=ctx.created_at,
        output_root=ctx.output_root,
        run_dir=ctx.run_dir,
        overwrite=ctx.overwrite,
        extra={
            "experiment": ctx.experiment,
            "attack_strength": ctx.attack_strength,
            "model_diversity": ctx.model_diversity,
        },
    )


def compute_descriptor_similarity_metrics(
    scenario_df: pd.DataFrame,
    edge_list: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> dict[str, float]:
    """Within/cross-model descriptor similarity for model-diversity experiment."""
    if scenario_df.empty:
        return {}
    scaler = resolve_fleet_scaler_from_config(config)
    X, _ = resolve_fleet_similarity_matrix(
        scenario_df,
        similarity_feature_view="behavior_only_vehicle_normalized",
        feature_dominance_threshold=5.0,
        allowed_high_dominance_features=frozenset(),
        fleet_scaler_provenance=scaler,
    )
    X = np.asarray(X, dtype=np.float64)
    meta = scenario_df.reset_index(drop=True)
    models = meta["vehicle_model"].astype(str).to_numpy()
    mal = meta["ground_truth_malicious"].to_numpy()
    camp = meta["ground_truth_campaign_member"].to_numpy()

    def _mean_sim(mask_a: np.ndarray, mask_b: np.ndarray | None = None) -> float:
        idx_a = np.where(mask_a)[0]
        if len(idx_a) < 2:
            return float("nan")
        if mask_b is None:
            pairs = []
            for i in range(len(idx_a)):
                for j in range(i + 1, len(idx_a)):
                    pairs.append(float(np.dot(X[idx_a[i]], X[idx_a[j]])))
            return float(np.mean(pairs)) if pairs else float("nan")
        idx_b = np.where(mask_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return float("nan")
        pairs = [float(np.dot(X[i], X[j])) for i in idx_a for j in idx_b]
        return float(np.mean(pairs)) if pairs else float("nan")

    within_model_mal = []
    cross_model_mal = []
    for model in np.unique(models):
        m_mask = (models == model) & (mal == 1)
        within_model_mal.append(_mean_sim(m_mask))
    mal_mask = mal == 1
    for i, m1 in enumerate(np.unique(models)):
        for m2 in np.unique(models)[i + 1 :]:
            cross_model_mal.append(
                _mean_sim((models == m1) & mal_mask, (models == m2) & mal_mask)
            )
    benign_cross = []
    for i, m1 in enumerate(np.unique(models)):
        for m2 in np.unique(models)[i + 1 :]:
            benign_cross.append(
                _mean_sim((models == m1) & (mal == 0), (models == m2) & (mal == 0))
            )

    cross_edges = same_edges = 0
    if not edge_list.empty and "source" in edge_list.columns and "target" in edge_list.columns:
        id_to_model = scenario_df.set_index("event_id")["vehicle_model"].to_dict()
        for _, e in edge_list.iterrows():
            m1 = id_to_model.get(e["source"])
            m2 = id_to_model.get(e["target"])
            if m1 and m2:
                if m1 != m2:
                    cross_edges += 1
                else:
                    same_edges += 1

    total_e = cross_edges + same_edges
    return {
        "within_model_attack_similarity": float(np.nanmean(within_model_mal)),
        "cross_model_attack_similarity": float(np.nanmean(cross_model_mal)),
        "campaign_similarity": _mean_sim(camp.astype(bool)),
        "benign_cross_model_similarity": float(np.nanmean(benign_cross)),
        "malicious_minus_benign_cross_sim": float(
            np.nanmean(cross_model_mal) - np.nanmean(benign_cross)
        ),
        "cross_model_edges": cross_edges,
        "same_model_edges": same_edges,
        "cross_model_edge_percentage": 100.0 * cross_edges / total_e if total_e else 0.0,
    }


def run_campaign_analysis_single(
    ctx: CampaignRunContext,
    *,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    model_composition: dict[str, int] | None = None,
    total_fleet_size: int = 20,
) -> dict[str, Any]:
    ensure_fleet_scaler_in_config(config, descriptors, manifest)
    t0 = time.perf_counter()
    scenario_df, mapping_df, membership = generate_campaign_analysis_scenario(
        experiment=ctx.experiment,  # type: ignore[arg-type]
        attack_strength=ctx.attack_strength,  # type: ignore[arg-type]
        seed=ctx.seed,
        descriptors=descriptors,
        manifest=manifest,
        catalog=catalog,
        config=config,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        model_composition=model_composition,
        total_fleet_size=total_fleet_size,
    )
    adapter = _adapter_ctx(ctx)
    if ctx.method == "local_ids":
        outputs = run_local_ids_method(adapter, scenario_df, membership, config)
    elif ctx.method == "descriptor_clustering":
        outputs = run_descriptor_clustering_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "standard_gnn":
        outputs = run_standard_gnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "fcgnn":
        outputs = run_fcgnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    else:
        raise ValueError(f"Unknown method: {ctx.method}")

    metrics = aggregate_run_metrics(
        method=ctx.method,
        seed=ctx.seed,
        scenario_key=f"{ctx.experiment}_{ctx.attack_strength}",
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        event_predictions=outputs.event_predictions,
        vehicle_predictions=outputs.vehicle_predictions,
        membership=membership,
        cluster_df=outputs.cluster_df,
        expect_campaign=True,
        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
    )
    metrics["experiment"] = ctx.experiment
    metrics["attack_strength"] = ctx.attack_strength
    metrics["attacked_model_diversity"] = int(scenario_df["model_diversity"].iloc[0])
    metrics["diversity_level"] = ctx.model_diversity
    metrics["total_fleet_size"] = total_fleet_size
    id_col = "vehicle_token" if "vehicle_token" in scenario_df.columns else "scenario_vehicle_id"
    metrics["attacked_vehicle_count"] = int(
        scenario_df.loc[scenario_df["ground_truth_campaign_member"] == 1, id_col].nunique()
    )

    sim_metrics = compute_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
    metrics.update(sim_metrics)

    if not outputs.graph_stats.empty:
        gs = outputs.graph_stats.iloc[0].to_dict()
        metrics.update({f"graph_{k}": v for k, v in gs.items() if k not in metrics})

    _write_outputs(ctx, scenario_df, mapping_df, membership, outputs, metrics, sim_metrics)
    return metrics


def _write_outputs(ctx, scenario_df, mapping_df, membership, outputs, metrics, sim_metrics) -> None:
    rd = ctx.run_dir
    scenario_df.to_csv(rd / "selected_source_records.csv", index=False)
    mapping_df.to_csv(rd / "scenario_vehicle_mapping.csv", index=False)
    membership.to_csv(rd / "vehicle_membership.csv", index=False)
    outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
    outputs.vehicle_predictions.to_csv(rd / "vehicle_predictions.csv", index=False)
    outputs.campaign_predictions.to_csv(rd / "campaign_predictions.csv", index=False)
    if not outputs.graph_stats.empty:
        outputs.graph_stats.to_csv(rd / "graph_statistics.csv", index=False)
    if not outputs.edge_list.empty:
        outputs.edge_list.to_csv(rd / "edge_list.csv", index=False)
    pd.DataFrame([sim_metrics]).to_csv(rd / "descriptor_similarity.csv", index=False)
    pd.DataFrame([metrics]).to_csv(rd / "run_level_metrics.csv", index=False)
    (rd / "runtime_memory.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


def run_campaign_analysis_corrected_single(
    ctx: CampaignRunContext,
    *,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    budget: DescriptorBudget,
    total_fleet_size: int = 20,
) -> dict[str, Any]:
    """Execute one corrected campaign-size run with fixed descriptor budget."""
    ensure_fleet_scaler_in_config(config, descriptors, manifest)
    t0 = time.perf_counter()
    scenario_df, mapping_df, membership, comp, budget = generate_corrected_campaign_scenario(
        attack_strength=ctx.attack_strength,  # type: ignore[arg-type]
        seed=ctx.seed,
        descriptors=descriptors,
        manifest=manifest,
        catalog=catalog,
        config=config,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        budget=budget,
    )
    adapter = _adapter_ctx(ctx)
    if ctx.method == "local_ids":
        outputs = run_local_ids_method(adapter, scenario_df, membership, config)
    elif ctx.method == "descriptor_clustering":
        outputs = run_descriptor_clustering_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "standard_gnn":
        outputs = run_standard_gnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "fcgnn":
        outputs = run_fcgnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    else:
        raise ValueError(f"Unknown method: {ctx.method}")

    metrics = aggregate_run_metrics(
        method=ctx.method,
        seed=ctx.seed,
        scenario_key=f"{ctx.experiment}_{ctx.attack_strength}",
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        event_predictions=outputs.event_predictions,
        vehicle_predictions=outputs.vehicle_predictions,
        membership=membership,
        cluster_df=outputs.cluster_df,
        expect_campaign=True,
        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
    )
    metrics["experiment"] = ctx.experiment
    metrics["attack_strength"] = ctx.attack_strength
    metrics["attacked_model_diversity"] = int(scenario_df["model_diversity"].iloc[0])
    metrics["total_fleet_size"] = total_fleet_size
    metrics["expected_total_nodes"] = budget.expected_total_nodes
    metrics["descriptors_per_vehicle"] = budget.descriptors_per_vehicle
    metrics["graph_nodes"] = len(scenario_df)
    id_col = "vehicle_token" if "vehicle_token" in scenario_df.columns else "scenario_vehicle_id"
    metrics["attacked_vehicle_count"] = int(
        scenario_df.loc[scenario_df["ground_truth_campaign_member"] == 1, id_col].nunique()
    )
    metrics["platform_composition"] = ",".join(f"{k}={v}" for k, v in sorted(comp.items()) if v > 0)

    sim_metrics = compute_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
    metrics.update(sim_metrics)
    if not outputs.graph_stats.empty:
        gs = outputs.graph_stats.iloc[0].to_dict()
        metrics.update({f"graph_{k}": v for k, v in gs.items() if k not in metrics})

    _write_outputs(ctx, scenario_df, mapping_df, membership, outputs, metrics, sim_metrics)
    rd = ctx.run_dir
    pd.DataFrame([build_vehicle_composition_row(
        ctx.run_id, ctx.seed, ctx.attack_strength, ctx.campaign_size, scenario_df, budget, comp,
    )]).to_csv(rd / "vehicle_composition.csv", index=False)
    pd.DataFrame([build_event_budget_validation_row(
        ctx.run_id, ctx.seed, ctx.campaign_size, scenario_df, budget,
    )]).to_csv(rd / "event_budget_validation.csv", index=False)
    return metrics
