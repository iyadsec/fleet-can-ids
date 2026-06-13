#!/usr/bin/env python3
"""Verify Phase 2 S0-S4 experiment completeness before Phase 3."""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.result_writer import ProtectedOutputGuard, load_experiment_config, verify_protected_directories_unchanged
from src.experiments.scenario_registry import SCENARIO_REGISTRY, enumerate_run_plan, resolve_method
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
METHODS = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]


def _scenario_hash(membership_path: Path) -> str:
    df = pd.read_csv(membership_path, usecols=["event_id"])
    payload = "|".join(sorted(df["event_id"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_phase2() -> tuple[list[str], list[str], dict]:
    project_root = resolve_project_root()
    guard = ProtectedOutputGuard(project_root, OUTPUT_ROOT)
    config = load_experiment_config(f"{OUTPUT_ROOT}/configs/final_validated_runs.yaml")
    critical: list[str] = []
    warnings: list[str] = []
    stats: dict = {}

    expected_plan = enumerate_run_plan(
        scenario_keys=list(SCENARIO_REGISTRY),
        methods=METHODS,
        seeds=REQUIRED_SEEDS,
        campaign_sizes=config.get("campaign", {}).get("campaign_sizes", [2, 5, 10]),
        coordination_strengths=config.get("campaign", {}).get("coordination_strengths", [0.0, 0.25, 0.75, 1.0]),
    )
    stats["expected_runs"] = len(expected_plan)

    manifest_path = guard.output_root / "manifests" / "results_manifest.csv"
    if not manifest_path.exists():
        critical.append(f"Missing manifest: {manifest_path}")
        return critical, warnings, stats

    manifest = pd.read_csv(manifest_path)
    completed = manifest[manifest["status"] == "completed"]
    failed = manifest[~manifest["status"].astype(str).str.startswith("completed")]
    stats["manifest_rows"] = len(manifest)
    stats["completed_rows"] = len(completed)
    stats["failed_rows"] = len(failed)

    if len(failed) > 0:
        critical.append(f"{len(failed)} non-completed manifest rows")
    if manifest["run_id"].duplicated().any():
        critical.append("Duplicate run_id values in results_manifest.csv")

    # Count on-disk completed runs (may exceed manifest if superseded dirs remain)
    on_disk = list((guard.output_root / "results").rglob("runs/*/run_level_metrics.csv"))
    stats["on_disk_runs"] = len(on_disk)
    manifest_run_ids = set(completed["run_id"].astype(str))
    extra_on_disk = stats["on_disk_runs"] - len(manifest_run_ids)
    if extra_on_disk > 0:
        warnings.append(
            f"{extra_on_disk} extra on-disk run directories not listed in manifest "
            "(likely superseded partial runs; manifest is authoritative)"
        )

    seeds_present = sorted(completed["seed"].unique().tolist()) if "seed" in completed.columns else []
    stats["seeds_present"] = seeds_present
    missing_seeds = [s for s in REQUIRED_SEEDS if s not in seeds_present]
    if missing_seeds:
        critical.append(f"Missing seeds in manifest: {missing_seeds}")

    # Per-scenario seed coverage
    for scenario_key in SCENARIO_REGISTRY:
        sub = completed[completed["scenario_key"] == scenario_key]
        if sub.empty:
            critical.append(f"No completed runs for {scenario_key}")
            continue
        n_seeds = sub["seed"].nunique()
        if n_seeds < len(REQUIRED_SEEDS):
            critical.append(f"{scenario_key}: only {n_seeds}/{len(REQUIRED_SEEDS)} seeds")

    # Scenario record identity per configuration (same membership hash across methods)
    def _resolve_run_dir(run_id: str) -> Path | None:
        for scenario_key in SCENARIO_REGISTRY:
            run_root = guard.output_root / "results" / scenario_key / "runs"
            if not run_root.exists():
                continue
            for run_dir in run_root.iterdir():
                if run_dir.name == run_id or run_dir.name.startswith(f"{run_id}_"):
                    return run_dir
        return None

    config_groups = completed.groupby(
        ["scenario_key", "seed", "campaign_size", "coordination_strength"], dropna=False
    )
    for keys, group in config_groups:
        hashes: dict[str, str] = {}
        for _, row in group.iterrows():
            run_dir = _resolve_run_dir(str(row["run_id"]))
            mem = (run_dir / "scenario_membership.csv") if run_dir else None
            if run_dir is None or mem is None or not mem.exists():
                critical.append(f"Missing scenario_membership for manifest run {row['run_id']}")
                continue
            hashes[str(row["method"])] = _scenario_hash(mem)
        if len(set(hashes.values())) > 1:
            scenario_key, seed, cs, coord = keys
            critical.append(
                f"{scenario_key} seed={seed} cs={cs} coord={coord}: methods used different "
                f"scenario records ({hashes})"
            )

    ok_legacy, legacy_errs = verify_protected_directories_unchanged(guard)
    if not ok_legacy:
        critical.extend(legacy_errs)

    return critical, warnings, stats


def write_report(critical: list[str], warnings: list[str], stats: dict) -> Path:
    path = resolve_project_root() / OUTPUT_ROOT / "validation" / "phase2_completion_check.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = len(critical) == 0
    lines = [
        "# Phase 2 Completion Check",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        "",
        "## Counts",
        "",
        f"- Expected configuration count: **{stats.get('expected_runs', 'n/a')}**",
        f"- Manifest completed rows: **{stats.get('completed_rows', 'n/a')}**",
        f"- On-disk run_level_metrics files: **{stats.get('on_disk_runs', 'n/a')}**",
        f"- Failed/partial manifest rows: **{stats.get('failed_rows', 0)}**",
        f"- Seeds present: `{stats.get('seeds_present', [])}`",
        "",
        "## Critical",
        "",
    ]
    lines.extend([f"- {c}" for c in critical] if critical else ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- None"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    critical, warnings, stats = validate_phase2()
    report = write_report(critical, warnings, stats)
    print(f"Phase 2 check → {report}")
    print(f"Critical: {len(critical)}, expected={stats.get('expected_runs')}, completed={stats.get('completed_rows')}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
