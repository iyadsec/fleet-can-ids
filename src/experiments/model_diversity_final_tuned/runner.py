"""Orchestrate tuned Phase 4: freeze, validate, search, test evaluate, outputs."""

from __future__ import annotations

import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.campaign_analysis_corrected import _build_membership
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.model_diversity.compositions import METHOD_TO_CONFIG, REQUIRED_SEEDS
from src.experiments.model_diversity_final.similarity import compute_cosine_descriptor_similarity_metrics, similarity_diagnostics_row
from src.experiments.model_diversity_final.connectivity import audit_cross_model_connectivity
from src.experiments.model_diversity_final_tuned.audit_docs import (
    recompute_provisional_false_campaigns,
    write_current_campaign_gate_audit,
    write_false_campaign_metric_definition,
)
from src.experiments.model_diversity_final_tuned.d2_analysis import write_d2_analysis
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown
from src.experiments.model_diversity_final_tuned.freeze import write_frozen_pipeline_audit
from src.experiments.model_diversity_final_tuned.gate_search import run_gate_search
from src.experiments.model_diversity_final_tuned.guard import ModelDiversityFinalTunedGuard
from src.experiments.model_diversity_final_tuned.outputs import generate_mdf_figures, generate_mdf_tables
from src.experiments.model_diversity_final_tuned.scenario_cache import build_scenario_cache
from src.experiments.model_diversity_final_tuned.statistics import run_statistical_families
from src.experiments.model_diversity_final_tuned.tuned_gate import TunedGateConfig
from src.experiments.model_diversity_final_tuned.tuned_gated_method import run_tuned_gated_graph_method
from src.experiments.model_diversity_final_tuned.validation_scenarios import (
    SCENARIO_TYPES,
    build_validation_scenarios,
    scenario_expect_campaign,
)
from src.experiments.result_writer import ExperimentRunContext
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

METHODS_FLEET = ("descriptor_clustering", "fcgnn")
METHODS_ALL = ("local_ids", "descriptor_clustering", "fcgnn")


def _merge_config(cfg: dict, source_root: Path) -> dict:
    from src.experiments.result_writer import load_experiment_config

    base_path = cfg.get("paths", {}).get("base_scenario_config", "new_experiments/final_validated_runs/configs/final_validated_runs.yaml")
    base = load_experiment_config(base_path)
    merged = {**base, **cfg}
    for k in ("local_ids", "graph", "gnn", "campaign"):
        merged[k] = {**base.get(k, {}), **cfg.get(k, {})}
    merged["paths"]["anomaly_descriptors"] = str(source_root / "descriptors/all_descriptors.csv")
    merged["fleet_normalisation"] = {"scaler_cache": str(source_root / "scalers/fleet_benign_scaler_final.json")}
    return merged


def _collect_provisional_run_metrics(source_root: Path) -> pd.DataFrame:
    rows = []
    for p in source_root.rglob("run_level_metrics.csv"):
        if "dry_" in str(p):
            continue
        df = pd.read_csv(p)
        df["run_id"] = p.parent.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_gate_selection_report(path: Path, meta: dict[str, Any], candidates: pd.DataFrame, gate: TunedGateConfig) -> None:
    lines = [
        "# Gate selection report",
        "",
        f"**Selected candidate:** `{candidates[candidates['selected']]['candidate_id'].iloc[0] if candidates['selected'].any() else 'none'}`",
        f"**Config hash:** `{gate.config_hash()}`",
        f"**Feasible candidate found:** {meta.get('feasible_found')}",
        f"**Test data used for selection:** {meta.get('test_data_used', False)}",
        "",
        "## Constraints",
        "",
        "- V0 false campaign alert rate ≤ 0.05",
        "- V1 campaign alert rate = 0",
        "- V2 incorrect merging rate ≤ 0.05",
        "- Mean benign vehicles included ≤ 1.0",
        "- Membership precision ≥ 0.80",
        "",
        "## Selected gate parameters",
        "",
        "```yaml",
        yaml.dump(gate.to_dict()),
        "```",
        "",
        f"## Candidates evaluated: {meta.get('n_candidates', 0)}",
        f"## Pareto-optimal count: {int(candidates['pareto_optimal'].sum()) if 'pareto_optimal' in candidates.columns else 0}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tuned_phase4(*, skip_gate_search: bool = False, skip_test_runs: bool = False) -> dict[str, Any]:
    root = resolve_project_root()
    guard = ModelDiversityFinalTunedGuard(root)
    guard.ensure_directory_tree()
    out = guard.output_root
    src = guard.source_root

    # 1. Freeze inputs
    hashes = write_frozen_pipeline_audit(src, out / "audit/frozen_pipeline_inputs.md")

    # Copy objective config
    obj_src = root / "configs/gate_selection_objective.yaml"
    obj_dst = out / "configs/gate_selection_objective.yaml"
    if obj_src.exists():
        shutil.copy(obj_src, obj_dst)
    else:
        obj_dst.write_text(yaml.dump({"primary_objective": "maximize_validation_campaign_f1"}), encoding="utf-8")

    cfg_path = src / "configs/phase4_model_diversity_final.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    config = _merge_config(cfg, src)
    descriptors = pd.read_csv(src / "descriptors/all_descriptors.csv")
    window_manifest = pd.read_csv(src / "manifests/final_window_split_manifest.csv")
    ensure_fleet_scaler_in_config(config, descriptors, window_manifest)

    # 2. Validation scenarios
    val_catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        window_manifest,
        weak_threshold=0.55,
        strong_threshold=0.80,
        min_windows_per_segment=10,
        target_split="validation",
    )
    val_manifest = build_validation_scenarios(
        descriptors, window_manifest, val_catalog, config,
        src / "results", out / "validation_scenarios",
    )
    val_manifest.to_csv(out / "validation_scenarios/validation_scenario_manifest.csv", index=False)

    # 3–4. Audits
    write_false_campaign_metric_definition(out / "audit/false_campaign_metric_definition.md")
    write_current_campaign_gate_audit(out / "audit/current_campaign_gate_audit.md")
    prov_recomp = recompute_provisional_false_campaigns(src / "results", out / "results/provisional_false_campaign_recomputation.csv")

    # 5–8. Gate search on validation caches
    caches = []
    passed = val_manifest[val_manifest["validation_passed"] == True]  # noqa: E712
    for _, row in passed.iterrows():
        run_id = row["validation_run_id"]
        scenario = row["scenario"]
        rec = pd.read_csv(out / "validation_scenarios" / f"{run_id}_records.csv")
        mem = pd.read_csv(out / "validation_scenarios" / f"{run_id}_membership.csv")
        for method in METHODS_FLEET:
            try:
                caches.append(build_scenario_cache(
                    rec, mem, config, int(row["validation_seed"]), method,
                    run_id=run_id, scenario=scenario, expect_campaign=scenario_expect_campaign(scenario),
                ))
            except Exception as exc:
                (out / "logs" / f"cache_fail_{run_id}_{method}.log").write_text(traceback.format_exc(), encoding="utf-8")

    if skip_gate_search and (out / "configs/final_selected_campaign_gate.yaml").exists():
        gate = TunedGateConfig.from_dict(yaml.safe_load((out / "configs/final_selected_campaign_gate.yaml").read_text()))
        candidates = pd.read_csv(out / "gate_search/all_gate_candidates.csv") if (out / "gate_search/all_gate_candidates.csv").exists() else pd.DataFrame()
        search_meta = {"feasible_found": True, "n_candidates": len(candidates), "test_data_used": False}
    else:
        search_caches = []
        seen: set[str] = set()
        for cache in caches:
            if cache.method != "fcgnn":
                continue
            if cache.scenario in seen:
                continue
            seen.add(cache.scenario)
            search_caches.append(cache)
        gate, candidates, search_meta = run_gate_search(
            search_caches, out / "configs/gate_selection_objective.yaml", out / "gate_search/all_gate_candidates.csv",
        )
        search_meta["search_cache_count"] = len(search_caches)
        search_meta["full_validation_cache_count"] = len(caches)
        gate_dict = gate.to_dict()
        gate_dict["config_hash"] = gate.config_hash()
        gate_dict["frozen_at"] = datetime.now(timezone.utc).isoformat()
        (out / "configs/final_selected_campaign_gate.yaml").write_text(yaml.dump(gate_dict), encoding="utf-8")
        _write_gate_selection_report(out / "gate_search/gate_selection_report.md", search_meta, candidates, gate)

    # 9. D2 analysis (provisional baseline)
    provisional_metrics = _collect_provisional_run_metrics(src)

    # 11–13. Test re-evaluation on frozen 150 production runs
    run_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    if not skip_test_runs:
        for run_dir in sorted(src.rglob("runs/final_md_*")):
            if not run_dir.is_dir() or "dry_" in run_dir.name:
                continue
            scenario_df = pd.read_csv(run_dir / "selected_source_records.csv")
            import re

            m = re.search(
                r"final_md_(strong|weak)_d(\d+)_(local_ids|descriptor_clustering|fcgnn)_seed(\d+)",
                run_dir.name,
            )
            if not m:
                continue
            strength, diversity_level, method, seed = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            membership = _build_membership(scenario_df, seed, 5, 1.0, strength)
            tuned_run_id = f"tuned_{run_dir.name}"
            tuned_dir = out / "results" / strength / "runs" / tuned_run_id
            guard.validate_write_path(tuned_dir)
            tuned_dir.mkdir(parents=True, exist_ok=True)

            try:
                if method == "local_ids":
                    ctx = ExperimentRunContext(
                        run_id=tuned_run_id, scenario_key=f"model_diversity_tuned_{strength}",
                        method=method, seed=seed, campaign_size=5, coordination_strength=1.0,
                        created_at=datetime.now(timezone.utc).isoformat(), output_root=out, run_dir=tuned_dir,
                    )
                    outputs = run_local_ids_method(ctx, scenario_df, membership, config)
                else:
                    outputs = run_tuned_gated_graph_method(scenario_df, membership, config, seed, method, gate)

                sim = compute_cosine_descriptor_similarity_metrics(scenario_df, outputs.edge_list, config=config)
                conn = audit_cross_model_connectivity(scenario_df, outputs.edge_list, outputs.event_predictions, run_id=tuned_run_id)
                fb = compute_false_campaign_breakdown(
                    outputs.event_predictions, membership, outputs.cluster_df, expect_campaign=True,
                )
                row = {
                    "run_id": tuned_run_id,
                    "provisional_run_id": run_dir.name,
                    "method": method,
                    "framework_config": METHOD_TO_CONFIG.get(method, method),
                    "seed": seed,
                    "attack_strength": strength,
                    "diversity_level": diversity_level,
                    "analysis_tier": scenario_df["analysis_tier"].iloc[0] if "analysis_tier" in scenario_df.columns else "",
                    "is_dry_test": False,
                    "graph_nodes": len(scenario_df),
                    **fb,
                    **sim,
                    **conn,
                    "campaign_metrics_na": method == "local_ids",
                }
                if method == "local_ids":
                    for k in ("campaign_precision", "campaign_recall", "campaign_f1", "false_campaign_alert_rate"):
                        row[k] = float("nan")
                run_rows.append(row)
                outputs.event_predictions.to_csv(tuned_dir / "event_predictions.csv", index=False)
                pd.DataFrame([row]).to_csv(tuned_dir / "run_level_metrics.csv", index=False)
            except Exception as exc:
                error_rows.append({"run_id": run_dir.name, "error": str(exc)})

    metrics_df = pd.DataFrame(run_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(out / "results/final_run_level_metrics.csv", index=False)
        fleet = metrics_df[metrics_df["method"].isin(METHODS_FLEET)].copy()
        fleet.to_csv(out / "results/final_campaign_membership.csv", index=False)
        err_cols = [
            "run_id", "method", "attack_strength", "diversity_level", "framework_config",
            "false_campaign_alert_indicator", "false_campaign_cluster_count", "benign_vehicles_included",
            "extra_cluster_count", "incorrect_merging", "fragmentation",
        ]
        fleet[[c for c in err_cols if c in fleet.columns]].to_csv(
            out / "results/final_campaign_error_breakdown.csv", index=False
        )
        fleet[fleet["analysis_tier"] == "controlled_same_attack"].to_csv(
            out / "results/final_controlled_same_attack.csv", index=False
        )
        fleet[(fleet["attack_strength"] == "strong") & (fleet["diversity_level"] == 3)].to_csv(
            out / "results/final_strong_diversity.csv", index=False
        )
        fleet[(fleet["attack_strength"] == "weak") & (fleet["diversity_level"] == 3)].to_csv(
            out / "results/final_weak_diversity.csv", index=False
        )
        sim_cols = [c for c in fleet.columns if "similarity" in c or c.startswith("cross_model")]
        if sim_cols:
            fleet[["run_id", "method", *sim_cols]].to_csv(out / "results/final_similarity_metrics.csv", index=False)
            fleet[["run_id", "method", *sim_cols]].to_csv(out / "results/final_connectivity_metrics.csv", index=False)
        if "runtime_graph_construction_sec" in metrics_df.columns:
            metrics_df[["run_id", "method", "runtime_graph_construction_sec"]].to_csv(
                out / "results/final_runtime_memory.csv", index=False
            )

    # Safety metrics from validation V0-V2
    safety_rows = []
    for scenario in ("V0", "V1", "V2"):
        sub = metrics_df  # placeholder — use validation evaluation
        for cache in caches:
            if cache.scenario != scenario:
                continue
            from src.experiments.model_diversity_final_tuned.scenario_cache import evaluate_gate_on_cache
            m = evaluate_gate_on_cache(cache, gate)
            safety_rows.append({"scenario": scenario, "method": cache.method, "validation_run_id": cache.run_id, **m})
    pd.DataFrame(safety_rows).to_csv(out / "results/final_safety_metrics.csv", index=False)

    # Aggregate outputs
    if not metrics_df.empty:
        fleet = metrics_df[metrics_df["method"].isin(METHODS_FLEET)]
        fleet.groupby(["attack_strength", "diversity_level", "framework_config"]).mean(numeric_only=True).reset_index().to_csv(
            out / "results/final_campaign_metrics.csv", index=False
        )
        stats = run_statistical_families(metrics_df)
        stats.to_csv(out / "results/final_statistical_tests.csv", index=False)
        ci_rows = []
        for col in ("campaign_f1", "false_campaign_alert_rate", "benign_vehicles_included"):
            if col not in fleet.columns:
                continue
            for cfg, g in fleet.groupby("framework_config"):
                vals = g[col].astype(float).dropna().to_numpy()
                if len(vals) >= 2:
                    from scipy import stats as sp_stats

                    mean = float(vals.mean())
                    se = float(sp_stats.sem(vals))
                    h = se * sp_stats.t.ppf(0.975, len(vals) - 1)
                    ci_rows.append({"framework_config": cfg, "metric": col, "mean": mean, "ci95_low": mean - h, "ci95_high": mean + h})
        pd.DataFrame(ci_rows).to_csv(out / "results/final_confidence_intervals.csv", index=False)
        write_d2_analysis(metrics_df, provisional_metrics, out / "audit/controlled_D2_low_f1_analysis.md")
        generate_mdf_tables(out, metrics_df, gate.to_dict(), stats, prov_recomp)
        generate_mdf_figures(out, metrics_df)

    pd.DataFrame(error_rows).to_csv(out / "results/excluded_runs.csv", index=False)

    # Comparison report
    comp_lines = [
        "# Provisional vs tuned Phase 4",
        "",
        "## Gate",
        f"- Provisional: default CampaignGateConfig (not validation-tuned)",
        f"- Tuned: `{gate.config_hash()}`",
        "",
        "## False campaign semantics",
        "- Provisional legacy rate ≈ 1.0 when qualifying clusters exist (metric bug)",
        "- Tuned uses decomposed A–D metrics",
        "",
    ]
    if not metrics_df.empty and not provisional_metrics.empty:
        for col in ("false_campaign_alert_rate", "benign_vehicles_included", "campaign_f1"):
            if col in metrics_df.columns and col in provisional_metrics.columns:
                comp_lines.append(f"- {col}: provisional mean={provisional_metrics[col].mean():.3f}, tuned mean={metrics_df[col].mean():.3f}")
    (out / "comparison/provisional_vs_tuned_phase4.md").write_text("\n".join(comp_lines), encoding="utf-8")

    summary = {
        "validation_scenarios": len(val_manifest),
        "overlap_with_test": int((val_manifest["overlap_with_test"] > 0).sum()),
        "gate_candidates": search_meta.get("n_candidates", 0),
        "selected_gate_hash": gate.config_hash(),
        "production_runs": len(run_rows),
        "feasible_gate": search_meta.get("feasible_found", False),
    }
    return summary
