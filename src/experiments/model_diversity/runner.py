"""Execute a single Phase 4 model-diversity run."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from src.experiments.campaign_analysis_corrected import (
    DescriptorBudget,
    build_event_budget_validation_row,
    build_vehicle_composition_row,
)
from src.experiments.campaign_analysis_runner import compute_descriptor_similarity_metrics
from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.result_writer import ExperimentRunContext
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config, resolve_fleet_scaler_from_config
from src.experiments.method_descriptor_clustering import run_descriptor_clustering_method
from src.experiments.method_fcgnn import run_fcgnn_method
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.model_diversity.compositions import METHOD_TO_CONFIG, composition_label
from src.experiments.model_diversity.generator import generate_model_diversity_scenario
from src.experiments.model_diversity.guard import ModelDiversityGuard
from src.experiments.model_diversity.privacy import check_privacy, privacy_check_row
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


@dataclass
class ModelDiversityRunContext:
    run_id: str
    attack_strength: str
    diversity_level: int
    method: str
    seed: int
    campaign_size: int
    coordination_strength: float
    model_composition: dict[str, int]
    composition_label: str
    created_at: str
    output_root: Path
    run_dir: Path
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        guard: ModelDiversityGuard,
        attack_strength: str,
        diversity_level: int,
        method: str,
        seed: int,
        model_composition: dict[str, int],
        composition_label_str: str,
        campaign_size: int = 5,
        coordination_strength: float = 1.0,
        overwrite: bool = False,
    ) -> ModelDiversityRunContext:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        uid = uuid4().hex[:8]
        run_id = (
            f"model_diversity_{attack_strength}_d{diversity_level}_{method}_"
            f"seed{seed}_{ts}_{uid}"
        )
        run_dir = guard.output_root / "results" / attack_strength / "runs" / run_id
        guard.validate_write_path(run_dir)
        if run_dir.exists() and not overwrite:
            from src.experiments.result_writer import RunAlreadyExistsError

            raise RunAlreadyExistsError(f"Run exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            attack_strength=attack_strength,
            diversity_level=diversity_level,
            method=method,
            seed=seed,
            campaign_size=campaign_size,
            coordination_strength=coordination_strength,
            model_composition=model_composition,
            composition_label=composition_label_str,
            created_at=datetime.now(timezone.utc).isoformat(),
            output_root=guard.output_root,
            run_dir=run_dir,
            overwrite=overwrite,
        )


def _adapter_ctx(ctx: ModelDiversityRunContext) -> ExperimentRunContext:
    return ExperimentRunContext(
        run_id=ctx.run_id,
        scenario_key=f"model_diversity_{ctx.attack_strength}",
        method=ctx.method,
        seed=ctx.seed,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        created_at=ctx.created_at,
        output_root=ctx.output_root,
        run_dir=ctx.run_dir,
        overwrite=ctx.overwrite,
        extra={
            "attack_strength": ctx.attack_strength,
            "diversity_level": ctx.diversity_level,
            "composition_label": ctx.composition_label,
        },
    )


def run_model_diversity_single(
    ctx: ModelDiversityRunContext,
    *,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    budget: DescriptorBudget,
) -> dict[str, Any]:
    ensure_fleet_scaler_in_config(config, descriptors, manifest)
    scaler = resolve_fleet_scaler_from_config(config)
    t0 = time.perf_counter()

    scenario_df, mapping_df, membership, comp, budget = generate_model_diversity_scenario(
        attack_strength=ctx.attack_strength,
        seed=ctx.seed,
        descriptors=descriptors,
        manifest=manifest,
        catalog=catalog,
        config=config,
        model_composition=ctx.model_composition,
        diversity_level=ctx.diversity_level,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        budget=budget,
    )

    X, feat_cols = resolve_fleet_similarity_matrix(
        scenario_df,
        similarity_feature_view=config.get("graph", {}).get(
            "similarity_feature_view", "behavior_only_vehicle_normalized"
        ),
        feature_dominance_threshold=5.0,
        allowed_high_dominance_features=frozenset(),
        fleet_scaler_provenance=scaler,
    )
    priv_errors = check_privacy(scenario_df, similarity_columns=feat_cols, graph_x_columns=feat_cols)
    if priv_errors:
        raise ValueError(f"Privacy check failed: {priv_errors[0]}")

    adapter = _adapter_ctx(ctx)
    if ctx.method == "local_ids":
        outputs = run_local_ids_method(adapter, scenario_df, membership, config)
    elif ctx.method == "descriptor_clustering":
        outputs = run_descriptor_clustering_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "fcgnn":
        outputs = run_fcgnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    else:
        raise ValueError(f"Unsupported method {ctx.method}")

    metrics = aggregate_run_metrics(
        method=ctx.method,
        seed=ctx.seed,
        scenario_key=f"model_diversity_{ctx.attack_strength}",
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        event_predictions=outputs.event_predictions,
        vehicle_predictions=outputs.vehicle_predictions,
        membership=membership,
        cluster_df=outputs.cluster_df,
        expect_campaign=True,
        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
    )

    attacked = scenario_df[scenario_df["ground_truth_campaign_member"] == 1]
    inst_counts = attacked.groupby("vehicle_model")["scenario_vehicle_id"].nunique().to_dict()

    metrics.update(
        {
            "experiment": "model_diversity",
            "attack_strength": ctx.attack_strength,
            "diversity_level": ctx.diversity_level,
            "framework_config": METHOD_TO_CONFIG[ctx.method],
            "configuration": METHOD_TO_CONFIG[ctx.method],
            "composition_label": ctx.composition_label,
            "configured_campaign_size": ctx.campaign_size,
            "measured_campaign_size": int(attacked["scenario_vehicle_id"].nunique()),
            "total_fleet_size": budget.total_fleet_size,
            "Hyundai_attacked_instances": int(inst_counts.get("Hyundai", 0)),
            "Kia_attacked_instances": int(inst_counts.get("Kia", 0)),
            "Chevrolet_attacked_instances": int(inst_counts.get("Chevrolet", 0)),
            "attacked_model_diversity": int(attacked["vehicle_model"].nunique()),
            "expected_total_nodes": budget.expected_total_nodes,
            "graph_nodes": len(scenario_df),
            "campaign_metrics_na": ctx.method == "local_ids",
            "validation_status": "pass",
            "source_snapshot": ctx.composition_label,
        }
    )

    sim_metrics = compute_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
    metrics.update(sim_metrics)
    metrics["campaign_similarity_gap"] = float(
        sim_metrics.get("cross_model_attack_similarity", float("nan"))
        - sim_metrics.get("benign_cross_model_similarity", float("nan"))
    )
    metrics["within_model_benign_similarity"] = sim_metrics.get("benign_cross_model_similarity")

    if not outputs.graph_stats.empty:
        gs = outputs.graph_stats.iloc[0].to_dict()
        metrics.update({f"graph_{k}": v for k, v in gs.items() if k not in metrics})

    _write_run_outputs(ctx, scenario_df, mapping_df, membership, outputs, metrics, sim_metrics, budget, comp)
    return metrics


def _write_run_outputs(ctx, scenario_df, mapping_df, membership, outputs, metrics, sim_metrics, budget, comp) -> None:
    rd = ctx.run_dir
    scenario_df.to_csv(rd / "selected_source_records.csv", index=False)
    mapping_df.to_csv(rd / "scenario_vehicle_mapping.csv", index=False)
    membership.to_csv(rd / "scenario_membership.csv", index=False)
    outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
    outputs.vehicle_predictions.to_csv(rd / "vehicle_predictions.csv", index=False)
    outputs.campaign_predictions.to_csv(rd / "campaign_predictions.csv", index=False)
    if not outputs.graph_stats.empty:
        outputs.graph_stats.to_csv(rd / "graph_statistics.csv", index=False)
    if not outputs.edge_list.empty:
        outputs.edge_list.to_csv(rd / "edge_list.csv", index=False)
    pd.DataFrame([sim_metrics]).to_csv(rd / "descriptor_similarity.csv", index=False)
    pd.DataFrame([metrics]).to_csv(rd / "run_level_metrics.csv", index=False)
    pd.DataFrame([build_vehicle_composition_row(
        ctx.run_id, ctx.seed, ctx.attack_strength, ctx.campaign_size, scenario_df, budget, comp,
    )]).to_csv(rd / "vehicle_composition.csv", index=False)
    pd.DataFrame([build_event_budget_validation_row(
        ctx.run_id, ctx.seed, ctx.campaign_size, scenario_df, budget,
    )]).to_csv(rd / "event_budget_validation.csv", index=False)
    pd.DataFrame([privacy_check_row(ctx.run_id, [])]).to_csv(rd / "privacy_check.csv", index=False)
    (rd / "runtime_memory.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
