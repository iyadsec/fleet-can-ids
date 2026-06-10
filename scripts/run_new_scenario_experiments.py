#!/usr/bin/env python3
"""Master runner for controlled fleet-aware CAN IDS scenario experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.aggregation import (  # noqa: E402
    export_scenario_tables,
    generate_figures,
    write_final_summary,
)
from src.experiments.data_splits import validate_no_split_leakage
from src.experiments.experiment_runner import run_single_experiment
from src.experiments.result_writer import (  # noqa: E402
    ExperimentRunContext,
    ProtectedOutputGuard,
    load_experiment_config,
    save_protection_baseline,
)
from src.experiments.scenario_generator import (  # noqa: E402
    ensure_split_manifest,
    load_descriptor_tables,
)
from src.experiments.scenario_registry import (  # noqa: E402
    SCENARIO_REGISTRY,
    enumerate_run_plan,
    get_scenario,
    resolve_method,
    validate_registry,
)
from src.utils.paths import resolve_project_root

DEFAULT_CONFIG = "new_experiments/configs/scenario_experiments.yaml"


def _git_commit_hash(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _parse_methods(raw: str | None, config: dict) -> list[str]:
    if raw:
        return [resolve_method(m.strip()) for m in raw.split(",") if m.strip()]
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run controlled fleet CAN IDS scenario experiments.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--all-scenarios", action="store_true")
    p.add_argument("--scenario", help="e.g. S4_weak_campaign")
    p.add_argument("--methods", help="local,clustering,standard_gnn,fcgnn")
    p.add_argument("--seeds", help="Comma-separated seeds")
    p.add_argument("--campaign-sizes", help="Comma-separated N attacked vehicles")
    p.add_argument("--coordination-strengths", help="Comma-separated strengths")
    p.add_argument("--reuse-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick-test", action="store_true")
    p.add_argument("--regenerate-tables", action="store_true")
    p.add_argument("--regenerate-figures", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--edge-sensitivity", action="store_true", help="Run M3/M4 edge sweep on S4")
    return p


def _resolve_list(args, config, quick_key, full_key, arg_val, cast):
    if arg_val:
        return [cast(x) for x in arg_val.split(",")]
    if args.quick_test:
        return [cast(x) for x in config.get("general", {}).get("quick_test", {}).get(quick_key, [])]
    return [cast(x) for x in config.get("campaign", {}).get(full_key, [])]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = resolve_project_root()
    config = load_experiment_config(args.config)
    if args.quick_test:
        config = {**config, "quick_test_epochs": 5}
        gnn = dict(config.get("gnn", {}))
        gnn["epochs"] = 5
        config["gnn"] = gnn

    guard = ProtectedOutputGuard(project_root, config.get("general", {}).get("output_root", "new_experiments"))
    guard.ensure_directory_tree()
    save_protection_baseline(guard)

    if validate_registry():
        print("Scenario registry invalid", file=sys.stderr)
        return 1

    if args.validate_only:
        from scripts.validate_new_scenario_results import main as validate_main
        return validate_main(["--config", str(args.config)])

    if args.regenerate_tables or args.regenerate_figures:
        for key in SCENARIO_REGISTRY:
            if args.regenerate_tables:
                export_scenario_tables(guard.output_root, key)
        if args.regenerate_figures:
            generate_figures(guard.output_root)
        write_final_summary(guard.output_root, args.config)
        return 0

    scenario_keys = list(SCENARIO_REGISTRY) if args.all_scenarios or not args.scenario else [args.scenario]
    for k in scenario_keys:
        get_scenario(k)

    methods = _parse_methods(args.methods, config)
    if args.quick_test and not args.methods:
        methods = [resolve_method(m) for m in config.get("general", {}).get("quick_test", {}).get("methods", methods)]

    seeds = _resolve_list(args, config, "seeds", "seeds", args.seeds, int) or [11]
    campaign_sizes = _resolve_list(args, config, "campaign_sizes", "campaign_sizes", args.campaign_sizes, int) or [2]
    coordination_strengths = _resolve_list(
        args, config, "coordination_strengths", "coordination_strengths", args.coordination_strengths, float
    ) or [0.0, 1.0]

    run_plan = enumerate_run_plan(
        scenario_keys=scenario_keys,
        methods=methods,
        seeds=seeds,
        campaign_sizes=campaign_sizes,
        coordination_strengths=coordination_strengths,
    )

    commit = _git_commit_hash(project_root)
    manifest_path = guard.output_root / "manifests" / "results_manifest.csv"
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.dry_run:
        with manifest_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "planned_at", "git_commit", "dry_run", "scenario_key", "scenario_id",
                    "method", "seed", "campaign_size", "coordination_strength", "status", "run_id",
                ],
            )
            w.writeheader()
            for entry in run_plan:
                w.writerow({**entry, "planned_at": timestamp, "git_commit": commit, "dry_run": True, "status": "dry_run_planned", "run_id": ""})
        print(f"Dry-run: {len(run_plan)} planned runs → {manifest_path}")
        from scripts.validate_new_scenario_results import main as validate_main
        return validate_main(["--config", str(args.config), "--phase", "1"])

    # Load data once
    paths = config.get("paths", {})
    descriptors, features = load_descriptor_tables(
        project_root / paths.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv"),
        project_root / paths.get("window_features", "data/processed/window_features.csv"),
    )
    splits = config.get("splits", {})
    manifest_path_split = guard.output_root / "manifests" / "split_manifest.csv"
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=manifest_path_split,
        seed=int(seeds[0]),
        train_ratio=float(splits.get("train", 0.7)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )
    leak_errs = validate_no_split_leakage(manifest, tuple(splits.get("group_by", ["source_file", "vehicle_model"])))
    if leak_errs:
        print("Split leakage:", leak_errs, file=sys.stderr)
        return 1

    overwrite = bool(config.get("general", {}).get("overwrite", False))
    completed = 0
    failed = 0
    rows: list[dict] = []

    edge_taus = config.get("graph", {}).get("similarity_thresholds", [0.95])
    edge_ks = config.get("graph", {}).get("max_neighbors", [10])

    for entry in run_plan:
        spec = get_scenario(entry["scenario_key"])
        method_id = entry["method"]
        tau_list = edge_taus if args.edge_sensitivity and method_id in ("standard_gnn", "fcgnn") else [None]
        k_list = edge_ks if args.edge_sensitivity and method_id in ("standard_gnn", "fcgnn") else [None]

        for tau in tau_list:
            for k in k_list:
                suffix = ""
                if tau is not None:
                    suffix = f"tau{tau:.2f}".replace(".", "p")
                if k is not None:
                    suffix += f"_k{k}"
                try:
                    ctx = ExperimentRunContext.create(
                        guard=guard,
                        scenario_key=entry["scenario_key"],
                        method=method_id,
                        seed=entry["seed"],
                        campaign_size=entry["campaign_size"],
                        coordination_strength=entry["coordination_strength"],
                        overwrite=overwrite,
                        suffix=suffix or None,
                    )
                except FileExistsError:
                    if args.reuse_existing:
                        rows.append({**entry, "scenario_id": spec.scenario_id, "status": "skipped_existing", "run_id": ""})
                        continue
                    raise

                if args.reuse_existing and (ctx.run_dir / "run_level_metrics.csv").exists():
                    rows.append({**entry, "scenario_id": spec.scenario_id, "status": "reused", "run_id": ctx.run_id})
                    completed += 1
                    continue

                try:
                    ctx.write_config_snapshot(config)
                    run_single_experiment(
                        ctx, spec, method_id, config, descriptors, manifest,
                        similarity_threshold=tau,
                        max_neighbors=k,
                    )
                    rows.append({**entry, "scenario_id": spec.scenario_id, "status": "completed", "run_id": ctx.run_id})
                    completed += 1
                except Exception as exc:
                    failed += 1
                    rows.append({**entry, "scenario_id": spec.scenario_id, "status": f"failed:{exc}", "run_id": ctx.run_id})
                    traceback.print_exc()

    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "planned_at", "git_commit", "scenario_key", "scenario_id", "method", "seed",
                "campaign_size", "coordination_strength", "status", "run_id",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({**r, "planned_at": timestamp, "git_commit": commit})

    for key in scenario_keys:
        export_scenario_tables(guard.output_root, key)
    generate_figures(guard.output_root)
    write_final_summary(guard.output_root, args.config)

    print(f"Completed {completed} runs, {failed} failed → {manifest_path}")
    from scripts.validate_new_scenario_results import main as validate_main
    return validate_main(["--config", str(args.config), "--phase", "9"]) if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
