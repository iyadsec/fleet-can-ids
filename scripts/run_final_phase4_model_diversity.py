#!/usr/bin/env python3
"""Final Phase 4: split-faithful retraining, campaign gate, heterogeneous benign fleet."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from src.experiments.campaign_analysis_corrected import DescriptorBudget, DEFAULT_BENIGN_PER_ATTACKED, DEFAULT_BENIGN_PER_BENIGN, DEFAULT_DESCRIPTORS_PER_VEHICLE, DEFAULT_FLEET_SIZE, DEFAULT_MALICIOUS_PER_ATTACKED
from src.experiments.model_diversity.compositions import REQUIRED_SEEDS
from src.experiments.model_diversity_corrected.benign_fleet import BENIGN_FLEET_COMPOSITION
from src.experiments.model_diversity_corrected.compositions import resolve_corrected_composition, supported_conditions
from src.experiments.model_diversity_corrected.generator import generate_corrected_model_diversity_scenario
from src.experiments.model_diversity_final.audit_docs import (
    write_campaign_decision_audit,
    write_false_campaign_analysis,
    write_similarity_audit,
)
from src.experiments.model_diversity_final.campaign_gate import CampaignGateConfig, search_campaign_gate
from src.experiments.model_diversity_final.connectivity import audit_cross_model_connectivity
from src.experiments.model_diversity_final.gated_method import run_gated_graph_method
from src.experiments.model_diversity_final.guard import ModelDiversityFinalGuard
from src.experiments.model_diversity_final.pipeline import retrain_local_pipeline
from src.experiments.model_diversity_final.similarity import compute_cosine_descriptor_similarity_metrics, similarity_diagnostics_row
from src.experiments.model_diversity_final.source_pools import build_final_source_pools
from src.experiments.model_diversity_final.split import build_final_split_manifest, write_split_integrity_report
from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.model_diversity.compositions import METHOD_TO_CONFIG
from src.experiments.result_writer import ExperimentRunContext, RunAlreadyExistsError, load_experiment_config
from src.experiments.scenario_generator import load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity_final")
CONFIG_PATH = OUT / "configs/phase4_model_diversity_final.yaml"
METHODS = ("local_ids", "descriptor_clustering", "fcgnn")


def _merge_config(cfg: dict) -> dict:
    base = load_experiment_config(cfg["paths"]["base_scenario_config"])
    merged = {**base, **cfg}
    for k in ("local_ids", "graph", "gnn", "campaign"):
        merged[k] = {**base.get(k, {}), **cfg.get(k, {})}
    return merged


def _run_single(
    *,
    guard: ModelDiversityFinalGuard,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict,
    gate: CampaignGateConfig,
    spec: dict,
    method: str,
    is_dry: bool,
) -> dict:
    budget = DescriptorBudget(DEFAULT_DESCRIPTORS_PER_VEHICLE, DEFAULT_MALICIOUS_PER_ATTACKED, DEFAULT_BENIGN_PER_ATTACKED, DEFAULT_BENIGN_PER_BENIGN, DEFAULT_FLEET_SIZE)
    desc_scenario = descriptors.drop(columns=["split"], errors="ignore")
    scenario_df, mapping_df, membership, _, _ = generate_corrected_model_diversity_scenario(
        attack_strength=spec["attack_strength"],
        seed=spec["seed"],
        descriptors=desc_scenario,
        manifest=manifest,
        catalog=catalog,
        config=config,
        model_composition=spec["model_composition"],
        diversity_level=spec["diversity_level"],
        analysis_tier=spec["analysis_tier"],
        budget=budget,
    )
    prefix = "dry_" if is_dry else ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{prefix}final_md_{spec['attack_strength']}_d{spec['diversity_level']}_{method}_seed{spec['seed']}_{ts}"
    run_dir = guard.output_root / "results" / spec["attack_strength"] / "runs" / run_id
    guard.validate_write_path(run_dir)
    if run_dir.exists():
        raise RunAlreadyExistsError(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = ExperimentRunContext(
        run_id=run_id, scenario_key=f"model_diversity_final_{spec['attack_strength']}",
        method=method, seed=spec["seed"], campaign_size=5, coordination_strength=1.0,
        created_at=datetime.now(timezone.utc).isoformat(), output_root=guard.output_root, run_dir=run_dir,
    )
    if method == "local_ids":
        outputs = run_local_ids_method(ctx, scenario_df, membership, config)
    else:
        outputs = run_gated_graph_method(scenario_df, membership, config, spec["seed"], method, gate)

    metrics = aggregate_run_metrics(
        method=method, seed=spec["seed"], scenario_key=ctx.scenario_key,
        campaign_size=5, coordination_strength=1.0,
        event_predictions=outputs.event_predictions, vehicle_predictions=outputs.vehicle_predictions,
        membership=membership, cluster_df=outputs.cluster_df, expect_campaign=True,
        runtime=outputs.runtime,
    )
    sim = compute_cosine_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
    diag = similarity_diagnostics_row(run_id, scenario_df, config)
    conn = audit_cross_model_connectivity(scenario_df, outputs.edge_list, outputs.event_predictions, run_id=run_id)
    metrics.update(
        {
            "experiment": "model_diversity_final",
            "attack_strength": spec["attack_strength"],
            "diversity_level": spec["diversity_level"],
            "analysis_tier": spec["analysis_tier"],
            "framework_config": METHOD_TO_CONFIG[method],
            "is_dry_test": is_dry,
            "graph_nodes": len(scenario_df),
            **sim,
            **conn,
        }
    )
    benign_fleet = scenario_df[scenario_df.ground_truth_campaign_member == 0].groupby("vehicle_model")["scenario_vehicle_id"].nunique().to_dict()
    for k in ("Hyundai", "Kia", "Chevrolet"):
        metrics[f"{k}_benign_instances"] = int(benign_fleet.get(k, 0))

    scenario_df.to_csv(run_dir / "selected_source_records.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "run_level_metrics.csv", index=False)
    pd.DataFrame([diag]).to_csv(run_dir / "similarity_diagnostics.csv", index=False)
    pd.DataFrame([conn]).to_csv(run_dir / "cross_model_connectivity.csv", index=False)
    outputs.event_predictions.to_csv(run_dir / "event_predictions.csv", index=False)
    return metrics


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-runs", action="store_true")
    p.add_argument("--pipeline-only", action="store_true")
    p.add_argument("--skip-pipeline", action="store_true")
    args = p.parse_args(argv)

    root = resolve_project_root()
    guard = ModelDiversityFinalGuard(root)
    guard.ensure_directory_tree()
    out = guard.output_root

    cfg = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    if not cfg:
        cfg = {"paths": {"window_features": "data/processed/window_features.csv", "anomaly_descriptors": "data/processed/anomaly_descriptors.csv", "base_scenario_config": "new_experiments/final_validated_runs/configs/final_validated_runs.yaml"}, "local_ids": {"weak_threshold": 0.55, "strong_threshold": 0.80}}
        (root / CONFIG_PATH).parent.mkdir(parents=True, exist_ok=True)
        (root / CONFIG_PATH).write_text(yaml.dump(cfg), encoding="utf-8")
    config = _merge_config(cfg)

    desc_path = out / "descriptors/all_descriptors.csv"
    if args.skip_pipeline and desc_path.exists():
        descriptors = pd.read_csv(desc_path)
        window_manifest = pd.read_csv(out / "manifests/final_window_split_manifest.csv")
        split_summary = {"passed": True, "errors": []}
    else:
        features_path = root / config["paths"]["window_features"]
        features = pd.read_csv(features_path)
        trace_df, window_manifest, split_summary = build_final_split_manifest(
            features_path,
            root / config["paths"]["anomaly_descriptors"],
            out / "manifests/final_split_manifest.csv",
            seed=42,
        )
        window_manifest.to_csv(out / "manifests/final_window_split_manifest.csv", index=False)
        write_split_integrity_report(trace_df, window_manifest, out / "audit/final_split_integrity.md")
        if not split_summary["passed"]:
            raise SystemExit(f"Split audit failed: {split_summary['errors']}")

        pipe = retrain_local_pipeline(features, window_manifest, out, seed=42)
        pipe["local_metrics"].to_csv(out / "results/local_ids_by_model.csv", index=False)
        descriptors = pipe["descriptors"]
        build_final_source_pools(descriptors, window_manifest, out)

        write_similarity_audit(out / "audit/similarity_metric_audit.md")
        write_campaign_decision_audit(out / "audit/campaign_decision_logic_audit.md")

        gate = CampaignGateConfig()
        gate_path = out / "configs/final_campaign_gate.yaml"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(yaml.dump(gate.to_dict()), encoding="utf-8")
        search_campaign_gate([], out / "validation_tuning/campaign_gate_search.csv")

    if args.pipeline_only:
        return 0

    gate = CampaignGateConfig(**yaml.safe_load((out / "configs/final_campaign_gate.yaml").read_text()))
    config["paths"]["anomaly_descriptors"] = str(out / "descriptors/all_descriptors.csv")
    config["fleet_normalisation"] = {"scaler_cache": str(out / "scalers/fleet_benign_scaler_final.json")}
    ensure_fleet_scaler_in_config(config, descriptors, window_manifest)

    seeds = [11] if args.dry_run else list(REQUIRED_SEEDS)
    specs = []
    for row in supported_conditions():
        if not row["supported"] or row["seed"] not in seeds:
            continue
        comp, label, tier, _ = resolve_corrected_composition(row["attack_strength"], row["diversity_level"], row["seed"])
        if comp is None:
            continue
        specs.append({**row, "model_composition": comp, "composition_label": label, "analysis_tier": tier})

    window_manifest = pd.read_csv(out / "manifests/final_window_split_manifest.csv")
    desc_no_split = descriptors.drop(columns=["split"], errors="ignore")
    catalog = build_instance_catalog(
        desc_no_split, window_manifest, weak_threshold=0.55, strong_threshold=0.80, min_windows_per_segment=10,
    )

    if not args.skip_runs:
        ok = fail = 0
        log = []
        for spec in specs:
            for method in METHODS:
                try:
                    _run_single(guard=guard, descriptors=descriptors, manifest=window_manifest, catalog=catalog, config=config, gate=gate, spec=spec, method=method, is_dry=args.dry_run)
                    ok += 1
                except RunAlreadyExistsError:
                    pass
                except Exception as exc:
                    fail += 1
                    log.append(f"FAIL {spec} {method}: {exc}\n{traceback.format_exc()}")
        (out / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log").write_text("\n".join(log), encoding="utf-8")
        print(f"Runs OK={ok} FAIL={fail}")
        if fail:
            return 1

    write_false_campaign_analysis(out / "audit/final_false_campaign_analysis.md")
    summary = f"""# Final Phase 4 model diversity summary

## Leakage
Split-faithful retraining: IF and scalers fit on train benign only; thresholds from validation.

## Benign fleet
{BENIGN_FLEET_COMPOSITION}

## Gate
See configs/final_campaign_gate.yaml

Runs: see results/
"""
    (out / "FINAL_PHASE4_MODEL_DIVERSITY_SUMMARY.md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
