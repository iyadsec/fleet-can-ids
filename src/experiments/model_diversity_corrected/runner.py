"""Execute corrected Phase 4 model-diversity runs."""

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
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config, resolve_fleet_scaler_from_config
from src.experiments.method_descriptor_clustering import run_descriptor_clustering_method
from src.experiments.method_fcgnn import run_fcgnn_method
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.model_diversity.compositions import METHOD_TO_CONFIG
from src.experiments.model_diversity_corrected.benign_fleet import BENIGN_FLEET_COMPOSITION
from src.experiments.model_diversity_corrected.generator import generate_corrected_model_diversity_scenario
from src.experiments.model_diversity_corrected.guard import ModelDiversityCorrectedGuard
from src.experiments.model_diversity.privacy import check_privacy, privacy_check_row
from src.experiments.result_writer import ExperimentRunContext
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix


@dataclass
class CorrectedRunContext:
    run_id: str
    attack_strength: str
    diversity_level: int
    analysis_tier: str
    method: str
    seed: int
    campaign_size: int
    coordination_strength: float
    model_composition: dict[str, int]
    composition_label: str
    created_at: str
    output_root: Path
    run_dir: Path
    is_dry_test: bool = False
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        guard: ModelDiversityCorrectedGuard,
        attack_strength: str,
        diversity_level: int,
        analysis_tier: str,
        method: str,
        seed: int,
        model_composition: dict[str, int],
        composition_label_str: str,
        campaign_size: int = 5,
        coordination_strength: float = 1.0,
        is_dry_test: bool = False,
        overwrite: bool = False,
    ) -> CorrectedRunContext:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        uid = uuid4().hex[:8]
        prefix = "dry_" if is_dry_test else ""
        run_id = (
            f"{prefix}model_diversity_corrected_{attack_strength}_d{diversity_level}_{method}_"
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
            analysis_tier=analysis_tier,
            method=method,
            seed=seed,
            campaign_size=campaign_size,
            coordination_strength=coordination_strength,
            model_composition=model_composition,
            composition_label=composition_label_str,
            created_at=datetime.now(timezone.utc).isoformat(),
            output_root=guard.output_root,
            run_dir=run_dir,
            is_dry_test=is_dry_test,
            overwrite=overwrite,
        )


def run_corrected_single(
    ctx: CorrectedRunContext,
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

    scenario_df, mapping_df, membership, comp, budget = generate_corrected_model_diversity_scenario(
        attack_strength=ctx.attack_strength,
        seed=ctx.seed,
        descriptors=descriptors,
        manifest=manifest,
        catalog=catalog,
        config=config,
        model_composition=ctx.model_composition,
        diversity_level=ctx.diversity_level,
        analysis_tier=ctx.analysis_tier,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        budget=budget,
    )

    _, feat_cols = resolve_fleet_similarity_matrix(
        scenario_df,
        similarity_feature_view=config.get("graph", {}).get("similarity_feature_view", "behavior_only_vehicle_normalized"),
        fleet_scaler_provenance=scaler,
    )
    priv_errors = check_privacy(scenario_df, similarity_columns=feat_cols, graph_x_columns=feat_cols)
    if priv_errors:
        raise ValueError(priv_errors[0])

    adapter = ExperimentRunContext(
        run_id=ctx.run_id,
        scenario_key=f"model_diversity_corrected_{ctx.attack_strength}",
        method=ctx.method,
        seed=ctx.seed,
        campaign_size=ctx.campaign_size,
        coordination_strength=ctx.coordination_strength,
        created_at=ctx.created_at,
        output_root=ctx.output_root,
        run_dir=ctx.run_dir,
        overwrite=ctx.overwrite,
    )
    if ctx.method == "local_ids":
        outputs = run_local_ids_method(adapter, scenario_df, membership, config)
    elif ctx.method == "descriptor_clustering":
        outputs = run_descriptor_clustering_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    elif ctx.method == "fcgnn":
        outputs = run_fcgnn_method(adapter, scenario_df, membership, config, seed=ctx.seed)
    else:
        raise ValueError(ctx.method)

    metrics = aggregate_run_metrics(
        method=ctx.method, seed=ctx.seed,
        scenario_key=f"model_diversity_corrected_{ctx.attack_strength}",
        campaign_size=ctx.campaign_size, coordination_strength=ctx.coordination_strength,
        event_predictions=outputs.event_predictions, vehicle_predictions=outputs.vehicle_predictions,
        membership=membership, cluster_df=outputs.cluster_df, expect_campaign=True,
        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
    )
    attacked = scenario_df[scenario_df.ground_truth_campaign_member == 1]
    inst = attacked.groupby("vehicle_model")["scenario_vehicle_id"].nunique().to_dict()
    benign_fleet = scenario_df[scenario_df.ground_truth_campaign_member == 0].groupby("vehicle_model")["scenario_vehicle_id"].nunique().to_dict()

    metrics.update(
        {
            "experiment": "model_diversity_corrected",
            "attack_strength": ctx.attack_strength,
            "diversity_level": ctx.diversity_level,
            "analysis_tier": ctx.analysis_tier,
            "framework_config": METHOD_TO_CONFIG[ctx.method],
            "configuration": METHOD_TO_CONFIG[ctx.method],
            "composition_label": ctx.composition_label,
            "benign_fleet_composition": ",".join(f"{k}={v}" for k, v in sorted(BENIGN_FLEET_COMPOSITION.items())),
            "Hyundai_attacked_instances": int(inst.get("Hyundai", 0)),
            "Kia_attacked_instances": int(inst.get("Kia", 0)),
            "Chevrolet_attacked_instances": int(inst.get("Chevrolet", 0)),
            "Hyundai_benign_instances": int(benign_fleet.get("Hyundai", 0)),
            "Kia_benign_instances": int(benign_fleet.get("Kia", 0)),
            "Chevrolet_benign_instances": int(benign_fleet.get("Chevrolet", 0)),
            "attacked_model_diversity": int(attacked.vehicle_model.nunique()),
            "benign_fleet_model_diversity": int(scenario_df[scenario_df.ground_truth_campaign_member == 0].vehicle_model.nunique()),
            "graph_nodes": len(scenario_df),
            "expected_total_nodes": budget.expected_total_nodes,
            "campaign_metrics_na": ctx.method == "local_ids",
            "is_dry_test": ctx.is_dry_test,
            "validation_status": "pass",
        }
    )
    sim = compute_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
    metrics.update(sim)
    metrics["campaign_similarity_gap"] = float(
        sim.get("cross_model_attack_similarity", float("nan")) - sim.get("benign_cross_model_similarity", float("nan"))
    )
    if not outputs.graph_stats.empty:
        metrics.update({f"graph_{k}": v for k, v in outputs.graph_stats.iloc[0].items() if k not in metrics})

    rd = ctx.run_dir
    scenario_df.to_csv(rd / "selected_source_records.csv", index=False)
    mapping_df.to_csv(rd / "scenario_vehicle_mapping.csv", index=False)
    membership.to_csv(rd / "scenario_membership.csv", index=False)
    outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
    pd.DataFrame([{"composition": BENIGN_FLEET_COMPOSITION, **benign_fleet}]).to_csv(rd / "benign_fleet_composition.csv", index=False)
    pd.DataFrame([metrics]).to_csv(rd / "run_level_metrics.csv", index=False)
    pd.DataFrame([sim]).to_csv(rd / "descriptor_similarity.csv", index=False)
    pd.DataFrame([privacy_check_row(ctx.run_id, [])]).to_csv(rd / "privacy_check.csv", index=False)
    (rd / "runtime_memory.json").write_text(json.dumps(metrics, default=str, indent=2), encoding="utf-8")
    return metrics
