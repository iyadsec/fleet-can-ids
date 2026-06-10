#!/usr/bin/env python3
"""Validate new_experiments/ outputs and protect legacy result directories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.result_writer import (  # noqa: E402
    PROTECTED_RELATIVE_DIRS,
    ProtectedOutputGuard,
    load_experiment_config,
    verify_protected_directories_unchanged,
)
from src.experiments.scenario_registry import SCENARIO_REGISTRY, validate_registry
from src.utils.paths import resolve_project_root

DEFAULT_CONFIG = "new_experiments/configs/scenario_experiments.yaml"


def _check_no_writes_outside_new_experiments(guard: ProtectedOutputGuard) -> list[str]:
    errors: list[str] = []
    manifest = guard.output_root / "manifests" / "results_manifest.csv"
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8")
        for protected in PROTECTED_RELATIVE_DIRS:
            needle = f"{protected}/"
            if needle in text and f"new_experiments/{protected}/" not in text:
                errors.append(f"Manifest references protected path pattern: {needle}")
    return errors


def _check_config_snapshot_exists(guard: ProtectedOutputGuard) -> list[str]:
    errors: list[str] = []
    runs_root = guard.output_root / "results"
    if not runs_root.exists():
        return errors
    for run_dir in runs_root.rglob("runs/*"):
        if run_dir.is_dir():
            snapshot = run_dir / "config_snapshot.yaml"
            if not snapshot.exists():
                errors.append(f"Missing config snapshot: {snapshot}")
    return errors


def _check_temporal_edges_disabled(config: dict) -> list[str]:
    graph = config.get("graph", {})
    if graph.get("use_temporal_edges", False):
        return ["graph.use_temporal_edges must be false"]
    return []


def _check_isolation_forest_benign_only(config: dict) -> list[str]:
    local = config.get("local_ids", {})
    if local.get("model") != "isolation_forest":
        return [f"local_ids.model must be isolation_forest (got {local.get('model')})"]
    if not local.get("benign_only_training", True):
        return ["local_ids.benign_only_training must be true"]
    return []


def _write_validation_report(
    guard: ProtectedOutputGuard,
    *,
    phase: int,
    checks: list[tuple[str, bool, str]],
    critical_failures: list[str],
) -> Path:
    report_path = guard.output_root / "validation" / "final_validation_report.md"
    guard.validate_write_path(report_path)
    lines = [
        "# New Scenario Experiments — Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Phase:** {phase}",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"| {name} | {status} | {detail} |")
    lines.extend(
        [
            "",
            "## Critical failures",
            "",
        ]
    )
    if critical_failures:
        lines.extend(f"- {e}" for e in critical_failures)
    else:
        lines.append("- None")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate new scenario experiment outputs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--phase", type=int, default=9, help="Validation phase (1=dry-run only)")
    args = parser.parse_args(argv)

    project_root = resolve_project_root()
    config = load_experiment_config(args.config)
    guard = ProtectedOutputGuard(
        project_root, config.get("general", {}).get("output_root", "new_experiments")
    )

    checks: list[tuple[str, bool, str]] = []
    critical: list[str] = []

    # 1. Registry consistency
    reg_errors = validate_registry()
    ok = len(reg_errors) == 0
    checks.append(("Scenario registry", ok, "; ".join(reg_errors) or "All S0–S4 specs valid"))

    # 2. Protected directories unchanged
    protected_ok, protected_errors = verify_protected_directories_unchanged(guard)
    checks.append(
        (
            "Protected legacy directories unchanged",
            protected_ok,
            "; ".join(protected_errors) or "Fingerprints match baseline",
        )
    )
    if not protected_ok:
        critical.extend(protected_errors)

    # 3. Config constraints
    for name, errs in [
        ("Temporal edges disabled", _check_temporal_edges_disabled(config)),
        ("Isolation Forest benign-only", _check_isolation_forest_benign_only(config)),
        ("No manifest refs to legacy paths", _check_no_writes_outside_new_experiments(guard)),
    ]:
        ok = len(errs) == 0
        checks.append((name, ok, "; ".join(errs) or "OK"))
        if not ok:
            critical.extend(errs)

    # 4. Config snapshots for completed runs
    snapshot_errs = _check_config_snapshot_exists(guard)
    has_run_dirs = any(
        (guard.output_root / "results" / sk / "runs").exists()
        for sk in SCENARIO_REGISTRY
    )
    if has_run_dirs:
        ok = len(snapshot_errs) == 0
        checks.append(("Run config snapshots", ok, "; ".join(snapshot_errs) or "OK"))
        if not ok and args.phase > 1:
            critical.extend(snapshot_errs)
    elif args.phase <= 1:
        checks.append(("Run config snapshots", True, "Skipped — no run dirs yet (manifest-only dry-run)"))
    else:
        ok = len(snapshot_errs) == 0
        checks.append(("Run config snapshots", ok, "; ".join(snapshot_errs) or "OK"))
        if not ok:
            critical.extend(snapshot_errs)

    # Phase 1: skip metric recomputation / figure checks
    if args.phase >= 9:
        manifest = guard.output_root / "manifests" / "results_manifest.csv"
        if not manifest.exists():
            critical.append("Missing results_manifest.csv")
            checks.append(("Results manifest exists", False, "Missing"))
        else:
            checks.append(("Results manifest exists", True, str(manifest)))

    report = _write_validation_report(guard, phase=args.phase, checks=checks, critical_failures=critical)
    print(f"Validation report → {report}")

    if critical:
        print("CRITICAL validation failures:", file=sys.stderr)
        for err in critical:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
