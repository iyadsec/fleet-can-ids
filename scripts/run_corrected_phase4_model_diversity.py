#!/usr/bin/env python3
"""Corrected Phase 4: heterogeneous benign fleet model-diversity experiment."""

from __future__ import annotations

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
from src.experiments.model_diversity.collect import build_confidence_intervals, collect_from_runs
from src.experiments.model_diversity.compositions import REQUIRED_SEEDS
from src.experiments.model_diversity.statistics import run_model_diversity_statistics
from src.experiments.model_diversity_corrected.audit import run_full_audit
from src.experiments.model_diversity_corrected.comparison import write_comparison_report
from src.experiments.model_diversity_corrected.compositions import resolve_corrected_composition, supported_conditions
from src.experiments.model_diversity_corrected.guard import ModelDiversityCorrectedGuard
from src.experiments.model_diversity_corrected.outputs import export_figures, export_tables
from src.experiments.model_diversity_corrected.runner import CorrectedRunContext, run_corrected_single
from src.experiments.result_writer import RunAlreadyExistsError, load_experiment_config
from src.experiments.scenario_generator import load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity_corrected")
ORIG = Path("new_experiments/final_validated_runs/model_diversity")
CONFIG = OUT / "configs/phase4_model_diversity_corrected.yaml"
METHODS = ("local_ids", "descriptor_clustering", "fcgnn")


def _merge_config(cfg: dict) -> dict:
    base = load_experiment_config(cfg["paths"]["base_scenario_config"])
    merged = {**base, **cfg}
    for k in ("local_ids", "graph", "gnn", "campaign"):
        merged[k] = {**base.get(k, {}), **cfg.get(k, {})}
    return merged


def _enumerate(seeds: list[int], dry: bool = False) -> list[dict]:
    rows = []
    for row in supported_conditions():
        if not row["supported"] or row["seed"] not in seeds:
            continue
        comp, label, tier, _ = resolve_corrected_composition(row["attack_strength"], row["diversity_level"], row["seed"])
        if comp is None:
            continue
        rows.append({**row, "model_composition": comp, "composition_label": label, "analysis_tier": tier, "is_dry_test": dry})
    return rows


def _merge_fleet_metadata(run_df: pd.DataFrame, fleet_df: pd.DataFrame) -> pd.DataFrame:
    if fleet_df.empty or run_df.empty:
        return fleet_df
    meta_cols = [c for c in ("run_id", "attack_strength", "diversity_level", "analysis_tier", "composition_label") if c in run_df.columns]
    if "run_id" not in meta_cols:
        return fleet_df
    merged = fleet_df.merge(run_df[meta_cols].drop_duplicates("run_id"), on="run_id", how="left", suffixes=("", "_run"))
    for col in ("attack_strength", "diversity_level", "analysis_tier"):
        if f"{col}_run" in merged.columns:
            merged[col] = merged[f"{col}_run"].combine_first(merged.get(col))
            merged.drop(columns=[f"{col}_run"], inplace=True)
    return merged


def _write_summary(
    output_root: Path,
    audit_summary: dict,
    run_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> None:
    c2 = fleet_df[fleet_df["framework_config"] == "C2"]
    c3 = fleet_df[fleet_df["framework_config"] == "C3"]
    ctrl = fleet_df[fleet_df.get("analysis_tier", "") == "controlled_same_attack"]
    ctrl_f1 = ctrl[(ctrl["framework_config"] == "C3") & (ctrl["attack_strength"] == "strong")]["campaign_f1"].mean() if not ctrl.empty else float("nan")
    c2_f1 = c2[c2["attack_strength"] == "strong"]["campaign_f1"].mean() if not c2.empty else float("nan")
    c3_f1 = c3[c3["attack_strength"] == "strong"]["campaign_f1"].mean() if not c3.empty else float("nan")
    c3_fcr = c3[c3["attack_strength"] == "strong"]["false_campaign_alert_rate"].mean() if not c3.empty else float("nan")
    text = f"""# Corrected Phase 4 model diversity summary

## 1. Why Kia and Chevrolet benign counts were previously zero
Global benign file shuffle in `build_split_manifest()` assigned all Kia/Chevrolet attack_free files to train only. Phase 4 audited test-split benign only.

## 2. Files/filters that caused the issue
- `src/experiments/data_splits.py` — `build_split_manifest()` global benign shuffle
- Phase 4 source-pool audit — test-split filter only

## 3. Corrected split required
Yes — per-model benign test reservation via `build_split_manifest_balanced_benign()`. Descriptors not regenerated; split designation changed for experiment pool.

## 4. Final benign counts by model (test split)
{audit_summary.get('corrected_test_benign', {})}

## 5. Final benign fleet composition
5 Hyundai + 5 Kia + 5 Chevrolet benign instances (fixed across all diversity levels).

## 6. Supported strong diversity levels
D1, D2, D3

## 7. Supported weak diversity levels
D1, D2 (D3 unsupported: no weak Chevrolet malicious descriptors)

## 8. Controlled same-attack Hyundai/Kia results
Strong C3 controlled mean F1: {ctrl_f1:.4f}

## 9. Exploratory three-platform results
D3 labelled exploratory_mixed_attack (Chevrolet fuzzy vs Hyundai/Kia malfunction).

## 10. Similarity-only vs GraphSAGE
Strong C2 mean F1: {c2_f1:.4f}
Strong C3 mean F1: {c3_f1:.4f}

## 11. False campaign findings
Strong C3 mean false campaign rate: {c3_fcr:.4f}

## 12. Descriptor portability
Mean campaign similarity gap: {sim_df['malicious_minus_benign_cross_sim'].mean() if not sim_df.empty and 'malicious_minus_benign_cross_sim' in sim_df.columns else 'N/A'}

## 13. Statistical significance
See `results/statistical_tests.csv` and Table DC9.

## 14. Runtime and memory
See `results/runtime_memory.csv` and Table DC8.

## 15. Safe for main paper
Tables DC2–DC4, DC7; Figures DC1–DC4; controlled Hyundai/Kia analysis (DC3).

## 16. Limitations
- Split relabelling without IF retraining
- Weak D3 unsupported
- D3 is exploratory mixed-attack, not pure same-attack diversity

## 17. Superseded original Phase 4 results
All fleet-correlation conclusions using homogeneous Hyundai-only benign background.

Completed runs (excl. dry): {len(run_df)}
"""
    (output_root / "CORRECTED_PHASE4_MODEL_DIVERSITY_SUMMARY.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seeds", default=None)
    p.add_argument("--skip-runs", action="store_true")
    args = p.parse_args(argv)

    project_root = resolve_project_root()
    guard = ModelDiversityCorrectedGuard(project_root)
    guard.ensure_directory_tree()
    out_root = guard.output_root

    cfg = yaml.safe_load((project_root / CONFIG).read_text(encoding="utf-8"))
    config = _merge_config(cfg)
    manifest, catalog, audit_summary = run_full_audit(config, out_root, project_root=project_root)
    pool = pd.read_csv(out_root / "manifests/model_diversity_source_pool_corrected.csv")
    print("Audit:", audit_summary["corrected_test_benign"])
    if not audit_summary["audit_passed"]:
        raise SystemExit("Audit failed: " + str(audit_summary.get("validation_errors")))
    if args.audit_only:
        return 0

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(REQUIRED_SEEDS)
    if args.dry_run:
        seeds = [seeds[0]]
    specs = _enumerate(seeds, dry=args.dry_run)
    print(f"Planned: {len(specs)} conditions × {len(METHODS)} methods = {len(specs) * len(METHODS)} runs")

    budget = DescriptorBudget(DEFAULT_DESCRIPTORS_PER_VEHICLE, DEFAULT_MALICIOUS_PER_ATTACKED, DEFAULT_BENIGN_PER_ATTACKED, DEFAULT_BENIGN_PER_BENIGN, DEFAULT_FLEET_SIZE)

    if not args.skip_runs:
        descriptors, features = load_descriptor_tables(project_root / config["paths"]["anomaly_descriptors"], project_root / config["paths"]["window_features"])
        local_cfg = config.get("local_ids", {})
        catalog = build_instance_catalog(
            descriptors, manifest,
            weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
            strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
            min_windows_per_segment=10,
        )
        log = []
        ok = fail = 0
        for spec in specs:
            for method in METHODS:
                try:
                    ctx = CorrectedRunContext.create(
                        guard=guard, attack_strength=spec["attack_strength"], diversity_level=spec["diversity_level"],
                        analysis_tier=spec["analysis_tier"], method=method, seed=spec["seed"],
                        model_composition=spec["model_composition"], composition_label_str=spec["composition_label"],
                        is_dry_test=spec.get("is_dry_test", False),
                    )
                    run_corrected_single(ctx, descriptors=descriptors, manifest=manifest, catalog=catalog, config=config, budget=budget)
                    ok += 1
                    log.append(f"OK {spec['attack_strength']} D{spec['diversity_level']} {method} seed{spec['seed']}")
                except RunAlreadyExistsError:
                    log.append(f"SKIP exists {spec} {method}")
                except Exception as exc:
                    fail += 1
                    log.append(f"FAIL {spec} {method}: {exc}\n{traceback.format_exc()}")
        (out_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log").write_text("\n".join(log), encoding="utf-8")
        print(f"Runs OK={ok} FAIL={fail}")
        if fail:
            return 1

    collected = collect_from_runs(out_root)
    run_df = collected["run_level_metrics"]
    if "is_dry_test" in run_df.columns:
        dry = run_df[run_df.is_dry_test == True]
        if not dry.empty:
            dry.to_csv(out_root / "results/excluded_runs.csv", index=False)
        run_df = run_df[run_df.is_dry_test != True]

    fleet_df = _merge_fleet_metadata(run_df, collected["fleet_campaign_metrics"])
    extra_cols = [c for c in ("analysis_tier", "benign_vehicles_incorrectly_included") if c in run_df.columns]
    if extra_cols and "run_id" in run_df.columns:
        fleet_df = fleet_df.merge(run_df[["run_id"] + extra_cols].drop_duplicates("run_id"), on="run_id", how="left", suffixes=("", "_run"))
        for col in extra_cols:
            if f"{col}_run" in fleet_df.columns:
                fleet_df[col] = fleet_df[f"{col}_run"].combine_first(fleet_df.get(col))
                fleet_df.drop(columns=[f"{col}_run"], inplace=True)

    sim_df = collected["descriptor_similarity"]
    if not sim_df.empty and "analysis_tier" in run_df.columns:
        sim_df = sim_df.merge(run_df[["run_id", "analysis_tier"]].drop_duplicates(), on="run_id", how="left")

    stats_df = run_model_diversity_statistics(fleet_df) if not fleet_df.empty else pd.DataFrame()
    ci_df = build_confidence_intervals(fleet_df) if not fleet_df.empty else pd.DataFrame()
    unsupported = pd.DataFrame(
        [{"attack_strength": r["attack_strength"], "diversity_level": r["diversity_level"], "reason": r["unsupported_reason"]}
         for r in supported_conditions() if not r["supported"]]
    ).drop_duplicates(subset=["attack_strength", "diversity_level"])

    results = out_root / "results"
    run_df.to_csv(results / "run_level_metrics.csv", index=False)
    for name, df in collected.items():
        if name == "run_level_metrics":
            continue
        if name == "fleet_campaign_metrics":
            fleet_df.to_csv(results / "fleet_campaign_metrics.csv", index=False)
        elif not df.empty:
            out_df = df
            if name in ("descriptor_similarity",) and "is_dry_test" in run_df.columns:
                dry_ids = set(run_df[run_df.is_dry_test == True]["run_id"]) if "run_id" in run_df.columns else set()
                if dry_ids and "run_id" in out_df.columns:
                    out_df = out_df[~out_df.run_id.isin(dry_ids)]
            out_df.to_csv(results / f"{name}.csv", index=False)

    if not stats_df.empty:
        stats_df.to_csv(results / "statistical_tests.csv", index=False)
    if not ci_df.empty:
        ci_df.to_csv(results / "confidence_intervals.csv", index=False)
    unsupported.to_csv(results / "unsupported_configurations.csv", index=False)

    orig_fleet = pd.read_csv(project_root / ORIG / "results/fleet_campaign_metrics.csv") if (project_root / ORIG / "results/fleet_campaign_metrics.csv").exists() else None
    export_tables(
        out_root, run_df=run_df, fleet_df=fleet_df,
        local_df=collected["local_event_metrics"],
        weak_df=collected["weak_campaign_support"],
        sim_df=sim_df, runtime_df=collected["runtime_memory"],
        stats_df=stats_df, unsupported_df=unsupported,
        audit_summary=audit_summary, pool_df=pool, orig_fleet_df=orig_fleet,
    )
    export_figures(out_root, fleet_df, sim_df, collected["runtime_memory"], orig_fleet_df=orig_fleet)
    if not run_df.empty and "run_id" in run_df.columns:
        run_df.groupby("run_id").first()[
            ["Hyundai_benign_instances", "Kia_benign_instances", "Chevrolet_benign_instances", "benign_fleet_composition"]
        ].reset_index().to_csv(results / "benign_fleet_composition.csv", index=False)
        graph_cols = [c for c in run_df.columns if c.startswith("graph_") or c in ("run_id", "attack_strength", "diversity_level", "framework_config", "seed")]
        if any(c.startswith("graph_") for c in graph_cols):
            run_df[graph_cols].to_csv(results / "graph_statistics.csv", index=False)
    write_comparison_report(project_root / ORIG, out_root, audit_summary=audit_summary, fleet_df=fleet_df, sim_df=sim_df, stats_df=stats_df)
    _write_summary(out_root, audit_summary, run_df, fleet_df, sim_df, stats_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
