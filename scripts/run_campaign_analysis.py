#!/usr/bin/env python3
"""Master runner for campaign-size and model-diversity sensitivity experiments."""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.campaign_analysis_audit import run_data_availability_audit
from src.experiments.campaign_analysis_outputs import (
    collect_run_metrics,
    export_experiment_a_tables,
    export_experiment_b_tables,
    generate_experiment_a_figures,
    generate_experiment_b_figures,
    write_final_summary,
)
from src.experiments.campaign_analysis_runner import run_campaign_analysis_single
from src.experiments.campaign_analysis_writer import (
    CampaignAnalysisGuard,
    CampaignRunContext,
    load_campaign_analysis_config,
)
from src.experiments.result_writer import RunAlreadyExistsError, load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.scenario_registry import resolve_method
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

DEFAULT_CONFIG = "new_experiments/campaign_analysis/configs/campaign_analysis.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run campaign analysis experiments.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--experiment", choices=["campaign_size", "model_diversity", "all"], default="all")
    p.add_argument("--attack-strength", choices=["strong", "weak"], default=None)
    p.add_argument("--methods", help="local,clustering,standard_gnn,fcgnn")
    p.add_argument("--seeds", help="Comma-separated seeds")
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--regenerate-tables", action="store_true")
    p.add_argument("--regenerate-figures", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--reuse-existing", action="store_true", help="Skip runs whose output directory already exists")
    return p


def _merge_config(campaign_cfg: dict) -> dict:
    base_path = campaign_cfg.get("paths", {}).get("base_scenario_config")
    if base_path:
        base = load_experiment_config(base_path)
        merged = {**base, **campaign_cfg}
        for key in ("local_ids", "graph", "gnn", "campaign"):
            merged[key] = {**base.get(key, {}), **campaign_cfg.get(key, {})}
        return merged
    return campaign_cfg


def _parse_methods(raw: str | None, config: dict, quick: bool) -> list[str]:
    if raw:
        return [resolve_method(m.strip()) for m in raw.split(",") if m.strip()]
    if quick:
        return [resolve_method(m) for m in config.get("general", {}).get("quick_test", {}).get("methods", [])]
    methods_cfg = config.get("methods", {})
    return [
        resolve_method(alias)
        for alias, enabled in [
            ("local", methods_cfg.get("local_ids", True)),
            ("clustering", methods_cfg.get("descriptor_clustering", True)),
            ("standard_gnn", methods_cfg.get("standard_gnn", True)),
            ("fcgnn", methods_cfg.get("fcgnn", True)),
        ]
        if enabled
    ]


def _parse_seeds(raw: str | None, config: dict, quick: bool) -> list[int]:
    if raw:
        return [int(s) for s in raw.split(",")]
    if quick:
        return [int(s) for s in config.get("general", {}).get("quick_test", {}).get("seeds", [11])]
    return [int(s) for s in config.get("general", {}).get("seeds", [11])]


def enumerate_runs(config: dict, experiment: str, quick: bool) -> list[dict]:
    runs: list[dict] = []
    exp_a = config.get("experiment_a", {})
    exp_b = config.get("experiment_b", {})
    sizes = exp_a.get("campaign_sizes", [2, 5, 10])
    if quick:
        sizes = config.get("general", {}).get("quick_test", {}).get("campaign_sizes", [2])
    strengths = exp_a.get("attack_strengths", ["strong", "weak"])
    coord = float(exp_a.get("coordination_strength", 1.0))
    fleet = int(exp_a.get("preferred_total_fleet_size", 20))

    if experiment in ("campaign_size", "all") and exp_a.get("enabled", True):
        for cs in sizes:
            for strength in strengths:
                if strength == "weak":
                    # Chevrolet has no weak events — still runnable on Hyundai/Kia pool
                    pass
                runs.append(
                    {
                        "experiment": "campaign_size",
                        "attack_strength": strength,
                        "campaign_size": int(cs),
                        "coordination_strength": coord,
                        "total_fleet_size": fleet,
                        "model_diversity": None,
                        "model_composition": None,
                    }
                )

    if experiment in ("model_diversity", "all") and exp_b.get("enabled", True):
        levels = exp_b.get("model_diversity_levels", [1, 2, 3])
        if quick:
            levels = config.get("general", {}).get("quick_test", {}).get("model_diversity_levels", [1])
        compositions = exp_b.get("compositions", {})
        fixed_cs = int(exp_b.get("fixed_campaign_size", 5))
        for level in levels:
            comp = compositions.get(f"diversity_{level}", {})
            for strength in exp_b.get("attack_strengths", ["strong", "weak"]):
                if strength == "weak" and level == 3:
                    continue  # Chevrolet excluded from weak
                runs.append(
                    {
                        "experiment": "model_diversity",
                        "attack_strength": strength,
                        "campaign_size": fixed_cs,
                        "coordination_strength": float(exp_b.get("coordination_strength", 1.0)),
                        "total_fleet_size": fleet,
                        "model_diversity": int(level),
                        "model_composition": comp,
                    }
                )
    return runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = resolve_project_root()
    campaign_cfg = load_campaign_analysis_config(args.config)
    config = _merge_config(campaign_cfg)
    output_root_rel = campaign_cfg.get("general", {}).get("output_root", "new_experiments/campaign_analysis")
    guard = CampaignAnalysisGuard(project_root, output_root_rel)
    guard.ensure_directory_tree()
    output_root = guard.output_root

    catalog, audit_summary = run_data_availability_audit(config, output_root)
    print(f"Audit complete: {len(catalog)} vehicle instances catalogued")

    if args.audit_only:
        return 0

    if args.validate_only:
        from scripts.validate_campaign_analysis import main as validate_main
        return validate_main(["--config", str(args.config)])

    descriptors, features = load_descriptor_tables(
        project_root / config["paths"]["anomaly_descriptors"],
        project_root / config["paths"]["window_features"],
    )
    splits = config.get("splits", {})
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=output_root / "manifests" / "split_manifest.csv",
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )

    methods = _parse_methods(args.methods, config, args.quick_test)
    seeds = _parse_seeds(args.seeds, config, args.quick_test)
    experiment = args.experiment
    runs = enumerate_runs(config, experiment, args.quick_test)
    if args.attack_strength:
        runs = [r for r in runs if r["attack_strength"] == args.attack_strength]

    total = len(runs) * len(methods) * len(seeds)
    print(f"Planned runs: {total} ({len(runs)} conditions × {len(methods)} methods × {len(seeds)} seeds)")

    if args.dry_run:
        for r in runs[:5]:
            print(f"  DRY: {r} × methods={methods} × seeds={seeds}")
        if len(runs) > 5:
            print(f"  ... and {len(runs)-5} more conditions")
        return 0

    if args.regenerate_tables or args.regenerate_figures:
        df_a = collect_run_metrics(output_root, "campaign_size")
        df_b = collect_run_metrics(output_root, "model_diversity")
        if args.regenerate_tables:
            export_experiment_a_tables(df_a, output_root)
            export_experiment_b_tables(df_b, output_root)
        if args.regenerate_figures:
            generate_experiment_a_figures(df_a, output_root)
            generate_experiment_b_figures(df_b, output_root)
        write_final_summary(output_root, audit_summary, df_a, df_b)
        return 0

    overwrite = bool(config.get("general", {}).get("overwrite", False))
    completed = failed = skipped = 0
    log_path = output_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"

    for run_spec in runs:
        for seed in seeds:
            for method in methods:
                try:
                    ctx = CampaignRunContext.create(
                        guard=guard,
                        experiment=run_spec["experiment"],
                        attack_strength=run_spec["attack_strength"],
                        method=method,
                        seed=seed,
                        campaign_size=run_spec["campaign_size"],
                        coordination_strength=run_spec["coordination_strength"],
                        model_diversity=run_spec.get("model_diversity"),
                        overwrite=overwrite,
                    )
                    run_campaign_analysis_single(
                        ctx,
                        descriptors=descriptors,
                        manifest=manifest,
                        catalog=catalog,
                        config=config,
                        model_composition=run_spec.get("model_composition"),
                        total_fleet_size=run_spec["total_fleet_size"],
                    )
                    completed += 1
                    msg = f"OK {run_spec['experiment']} {method} seed={seed} cs={run_spec['campaign_size']}"
                    print(msg)
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                except RunAlreadyExistsError:
                    if args.reuse_existing:
                        skipped += 1
                        continue
                    raise
                except Exception as exc:
                    failed += 1
                    print(f"FAIL {run_spec} {method} seed={seed}: {exc}", file=sys.stderr)
                    with log_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"FAIL {run_spec} {method} seed={seed}: {exc}\n")
                        fh.write(traceback.format_exc() + "\n")

    df_a = collect_run_metrics(output_root, "campaign_size")
    df_b = collect_run_metrics(output_root, "model_diversity")
    export_experiment_a_tables(df_a, output_root)
    export_experiment_b_tables(df_b, output_root)
    generate_experiment_a_figures(df_a, output_root)
    generate_experiment_b_figures(df_b, output_root)
    write_final_summary(output_root, audit_summary, df_a, df_b)

    print(f"Completed {completed}, failed {failed}, skipped {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
