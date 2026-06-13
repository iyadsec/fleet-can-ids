#!/usr/bin/env python3
"""Phase 4: vehicle-model diversity experiment."""

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
from src.experiments.campaign_analysis_writer import load_campaign_analysis_config
from src.experiments.model_diversity.audit import run_model_diversity_audit
from src.experiments.model_diversity.collect import build_confidence_intervals, collect_from_runs
from src.experiments.model_diversity.compositions import REQUIRED_SEEDS, resolve_composition, supported_conditions
from src.experiments.model_diversity.guard import ModelDiversityGuard
from src.experiments.model_diversity.outputs import export_figures, export_tables
from src.experiments.model_diversity.runner import ModelDiversityRunContext, run_model_diversity_single
from src.experiments.model_diversity.statistics import run_model_diversity_statistics
from src.experiments.result_writer import RunAlreadyExistsError, load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.utils.paths import resolve_project_root

OUT_ROOT = Path("new_experiments/final_validated_runs/model_diversity")
CONFIG_PATH = OUT_ROOT / "configs" / "phase4_model_diversity.yaml"
HIER_VALIDATION = Path("new_experiments/final_validated_runs/hierarchical_alignment/validation/hierarchical_alignment_validation.md")
METHODS = ("local_ids", "descriptor_clustering", "fcgnn")


def _merge_config(cfg: dict) -> dict:
    base_path = cfg.get("paths", {}).get("base_scenario_config")
    base = load_experiment_config(base_path) if base_path else {}
    merged = {**base, **cfg}
    for key in ("local_ids", "graph", "gnn", "campaign"):
        merged[key] = {**base.get(key, {}), **cfg.get(key, {})}
    return merged


def _gate_hierarchical() -> None:
    if not HIER_VALIDATION.exists():
        raise RuntimeError("Hierarchical alignment validation missing")
    text = HIER_VALIDATION.read_text(encoding="utf-8")
    if "Critical failures: 0" not in text or "**Result:** PASS" not in text:
        raise RuntimeError("Hierarchical alignment gate failed")


def _enumerate_runs(seeds: list[int]) -> list[dict]:
    runs = []
    for row in supported_conditions():
        if not row["supported"]:
            continue
        if row["seed"] not in seeds:
            continue
        comp, label, _ = resolve_composition(row["attack_strength"], row["diversity_level"], row["seed"])
        if comp is None:
            continue
        runs.append(
            {
                "attack_strength": row["attack_strength"],
                "diversity_level": row["diversity_level"],
                "seed": row["seed"],
                "model_composition": comp,
                "composition_label": label,
            }
        )
    return runs


def _write_summary(output_root: Path, audit_summary: dict, run_df: pd.DataFrame, fleet_df: pd.DataFrame, sim_df: pd.DataFrame, stats_df: pd.DataFrame) -> None:
    c2 = fleet_df[fleet_df["framework_config"] == "C2"]
    c3 = fleet_df[fleet_df["framework_config"] == "C3"]
    text = f"""# Phase 4 model diversity summary

## 1. Strong diversity levels supported
D{audit_summary.get('supported_strong_diversity_levels', [])}

## 2. Weak diversity levels supported
D{audit_summary.get('supported_weak_diversity_levels', [])} (D3 unsupported: no weak Chevrolet descriptors)

## 3. Local Isolation Forest stability
Strong mean F1: {run_df[run_df['attack_strength']=='strong'].groupby('framework_config')['f1'].mean().to_dict() if 'f1' in run_df.columns else 'see local_event_metrics.csv'}
Weak mean F1: {run_df[run_df['attack_strength']=='weak'].groupby('framework_config')['f1'].mean().to_dict() if 'f1' in run_df.columns else 'see local_event_metrics.csv'}

## 4. Cross-model malicious vs benign similarity
Mean campaign similarity gap: {sim_df['malicious_minus_benign_cross_sim'].mean() if not sim_df.empty and 'malicious_minus_benign_cross_sim' in sim_df.columns else 'N/A'}

## 5–6. GraphSAGE campaign correlation vs similarity-only
Strong C2 mean F1: {c2[c2['attack_strength']=='strong']['campaign_f1'].mean():.4f}
Strong C3 mean F1: {c3[c3['attack_strength']=='strong']['campaign_f1'].mean():.4f}

## 7. Cross-model edges
Mean cross-model edge % by diversity: {sim_df.groupby('diversity_level')['cross_model_edge_percentage'].mean().to_dict() if not sim_df.empty else {}}

## 8. False campaign alerts
Mean false campaign rate (C3 strong): {c3[c3['attack_strength']=='strong']['false_campaign_alert_rate'].mean():.4f}

## 9. Computational cost
See `results/runtime_memory.csv` and Table D6.

## 10. Dataset limitations
- Chevrolet has no weak malicious descriptors in held-out test split
- Weak D3 marked unsupported_by_dataset
- Attack families differ by platform (malfunction vs fuzzy)

## 11. Main paper
Tables D2–D4, D7; Figures D1–D4

## 12. Supplementary
Table D5–D6, Figure D5–D6, unsupported configurations

Completed runs: {len(run_df)}
"""
    (output_root / "PHASE4_MODEL_DIVERSITY_SUMMARY.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seeds", default=None, help="Comma-separated seeds (default: all 10)")
    p.add_argument("--attack-strength", choices=["strong", "weak"], default=None)
    p.add_argument("--skip-runs", action="store_true", help="Only aggregate existing runs")
    args = p.parse_args(argv)

    project_root = resolve_project_root()
    _gate_hierarchical()

    guard = ModelDiversityGuard(project_root)
    guard.ensure_directory_tree()
    output_root = guard.output_root

    cfg = yaml.safe_load((project_root / CONFIG_PATH).read_text(encoding="utf-8"))
    config = _merge_config(cfg)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(REQUIRED_SEEDS)

    catalog, pool, audit_summary = run_model_diversity_audit(config, output_root)
    print(f"Audit passed. Source pool rows: {len(pool)}")

    if args.audit_only:
        return 0

    runs = _enumerate_runs(seeds)
    if args.attack_strength:
        runs = [r for r in runs if r["attack_strength"] == args.attack_strength]
    print(f"Planned conditions: {len(runs)} × {len(METHODS)} methods = {len(runs)*len(METHODS)} runs")

    if args.dry_run:
        for r in runs[:5]:
            print(f"  DRY {r}")
        return 0

    budget = DescriptorBudget(
        DEFAULT_DESCRIPTORS_PER_VEHICLE,
        DEFAULT_MALICIOUS_PER_ATTACKED,
        DEFAULT_BENIGN_PER_ATTACKED,
        DEFAULT_BENIGN_PER_BENIGN,
        DEFAULT_FLEET_SIZE,
    )

    if not args.skip_runs:
        descriptors, features = load_descriptor_tables(
            project_root / config["paths"]["anomaly_descriptors"],
            project_root / config["paths"]["window_features"],
        )
        manifest = ensure_split_manifest(
            descriptors, features,
            output_path=output_root / "manifests" / "split_manifest.csv",
            seed=int(config.get("splits", {}).get("seed", 42)),
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        )
        log = []
        ok = fail = 0
        for spec in runs:
            for method in METHODS:
                try:
                    ctx = ModelDiversityRunContext.create(
                        guard=guard,
                        attack_strength=spec["attack_strength"],
                        diversity_level=spec["diversity_level"],
                        method=method,
                        seed=spec["seed"],
                        model_composition=spec["model_composition"],
                        composition_label_str=spec["composition_label"],
                    )
                    run_model_diversity_single(
                        ctx, descriptors=descriptors, manifest=manifest,
                        catalog=catalog, config=config, budget=budget,
                    )
                    ok += 1
                    log.append(f"OK {spec} {method}")
                except RunAlreadyExistsError:
                    log.append(f"SKIP exists {spec} {method}")
                except Exception as exc:
                    fail += 1
                    log.append(f"FAIL {spec} {method}: {exc}\n{traceback.format_exc()}")
        (output_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log").write_text("\n".join(log), encoding="utf-8")
        print(f"Runs OK={ok} FAIL={fail}")
        if fail:
            return 1

    collected = collect_from_runs(output_root)
    run_df = collected["run_level_metrics"]
    fleet_df = collected["fleet_campaign_metrics"]
    if not fleet_df.empty:
        fleet_df["attack_strength"] = fleet_df["run_id"].map(
            run_df.set_index("run_id")["attack_strength"].to_dict() if "run_id" in run_df.columns else {}
        )
        # merge attack_strength from run_df by run_id
        if "run_id" not in run_df.columns and not run_df.empty:
            run_df = run_df.copy()
            run_df["run_id"] = run_df.index.astype(str)
    if not fleet_df.empty and "attack_strength" not in fleet_df.columns:
        fleet_df = fleet_df.merge(run_df[["run_id", "attack_strength", "diversity_level"]], on="run_id", how="left")

    stats_df = run_model_diversity_statistics(fleet_df)
    ci_df = build_confidence_intervals(fleet_df)
    unsupported = pd.DataFrame(
        [{"attack_strength": r["attack_strength"], "diversity_level": r["diversity_level"], "reason": r["unsupported_reason"]}
         for r in supported_conditions() if not r["supported"]]
    ).drop_duplicates(subset=["attack_strength", "diversity_level"])

    results = output_root / "results"
    run_df.to_csv(results / "run_level_metrics.csv", index=False)
    if not run_df.empty:
        run_df[run_df["attack_strength"] == "strong"].to_csv(results / "strong_run_level_metrics.csv", index=False)
        run_df[run_df["attack_strength"] == "weak"].to_csv(results / "weak_run_level_metrics.csv", index=False)
    for name, df in collected.items():
        if name != "run_level_metrics" and not df.empty:
            df.to_csv(results / f"{name}.csv", index=False)
    stats_df.to_csv(results / "statistical_tests.csv", index=False)
    ci_df.to_csv(results / "confidence_intervals.csv", index=False)
    if not collected.get("campaign_membership", pd.DataFrame()).empty:
        collected["campaign_membership"].to_csv(results / "campaign_membership.csv", index=False)
    unsupported.to_csv(results / "unsupported_configurations.csv", index=False)

    export_tables(
        output_root,
        run_df=run_df,
        fleet_df=fleet_df,
        local_df=collected["local_event_metrics"],
        weak_df=collected["weak_campaign_support"],
        sim_df=collected["descriptor_similarity"],
        runtime_df=collected["runtime_memory"],
        stats_df=stats_df,
        unsupported_df=unsupported,
        audit_summary=audit_summary,
    )
    export_figures(output_root, fleet_df, collected["weak_campaign_support"], collected["descriptor_similarity"], collected["runtime_memory"])
    _write_summary(output_root, audit_summary, run_df, fleet_df, collected["descriptor_similarity"], stats_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
