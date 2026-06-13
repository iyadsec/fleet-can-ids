#!/usr/bin/env python3
"""Validate Phase 1 quick end-to-end test for final validated runs."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.local_descriptor_normalisation import load_scaler_provenance
from src.experiments.publication_manifest import _validate_scenario_semantics
from src.experiments.result_writer import (
    ProtectedOutputGuard,
    load_experiment_config,
    verify_protected_directories_unchanged,
)
from src.experiments.scenario_registry import SCENARIO_REGISTRY, get_scenario
from src.experiments.vehicle_identity import count_attacked_vehicle_instances
from src.utils.paths import resolve_project_root

QUICK_ROOT = "new_experiments/final_validated_runs/quick_test"
MAIN_CONFIG = "new_experiments/final_validated_runs/configs/final_validated_quick_test.yaml"
REQUIRED_METHODS = ("local_ids", "descriptor_clustering", "standard_gnn", "fcgnn")
REQUIRED_SEED = 11


def _scenario_hash(membership: pd.DataFrame) -> str:
    payload = "|".join(sorted(membership["event_id"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_quick_test() -> tuple[list[str], list[str]]:
    project_root = resolve_project_root()
    quick_guard = ProtectedOutputGuard(project_root, QUICK_ROOT)
    config = load_experiment_config(MAIN_CONFIG)
    critical: list[str] = []
    warnings: list[str] = []

    baseline = project_root / "new_experiments/final_validated_runs/validation/protected_dirs_baseline.json"
    ok_legacy, legacy_errs = verify_protected_directories_unchanged(
        ProtectedOutputGuard(project_root, "new_experiments/final_validated_runs"),
        baseline_path=baseline if baseline.exists() else None,
    )
    if not ok_legacy:
        critical.extend(legacy_errs)

    scaler_path = project_root / config.get("fleet_normalisation", {}).get(
        "scaler_cache", "new_experiments/metadata_correction/manifests/fleet_benign_scaler.json"
    )
    if not scaler_path.exists():
        critical.append(f"Fleet scaler missing: {scaler_path}")
    else:
        prov = load_scaler_provenance(scaler_path)
        if prov.attack_labels_used:
            critical.append("Fleet scaler used attack labels")
        if prov.training_split != "train":
            critical.append(f"Fleet scaler fitted on split={prov.training_split}")

    results_root = quick_guard.output_root / "results"
    if not results_root.exists():
        critical.append(f"No quick_test results at {results_root}")
        return critical, warnings

    for scenario_key in SCENARIO_REGISTRY:
        spec = get_scenario(scenario_key)
        run_root = results_root / scenario_key / "runs"
        if not run_root.exists():
            critical.append(f"Missing runs for {scenario_key}")
            continue

        seed_runs: dict[str, list[Path]] = {}
        for run_dir in run_root.iterdir():
            if not run_dir.is_dir():
                continue
            if f"seed{REQUIRED_SEED}" not in run_dir.name:
                continue
            method = None
            for m in REQUIRED_METHODS:
                if f"_{m}_" in run_dir.name:
                    method = m
                    break
            if method is None:
                warnings.append(f"Unrecognised method in {run_dir.name}")
                continue
            seed_runs.setdefault(method, []).append(run_dir)

        for method in REQUIRED_METHODS:
            if method not in seed_runs:
                critical.append(f"{scenario_key} missing method {method} seed {REQUIRED_SEED}")

        if len(seed_runs) < 4:
            continue

        hashes: list[str] = []
        memberships: list[pd.DataFrame] = []
        for method, dirs in seed_runs.items():
            run_dir = sorted(dirs, key=lambda p: p.name)[-1]
            mem_path = run_dir / "scenario_membership.csv"
            if not mem_path.exists():
                critical.append(f"Missing membership: {mem_path}")
                continue
            mem = pd.read_csv(mem_path)
            memberships.append(mem)
            hashes.append(_scenario_hash(mem))

        if len(set(hashes)) > 1:
            critical.append(f"{scenario_key}: methods used different scenario records (hash mismatch)")

        if not memberships:
            continue
        mem = memberships[0]
        metrics_path = sorted(seed_runs["local_ids"])[-1] / "run_level_metrics.csv"
        metrics = {}
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()

        sem_errors = _validate_scenario_semantics(scenario_key, mem, mem, metrics)
        for err in sem_errors:
            critical.append(f"{scenario_key}: {err}")

        if spec.scenario_id == "S0":
            if int(mem["ground_truth_malicious"].sum()) > 0:
                critical.append("S0 has malicious events")
        elif spec.scenario_id == "S1":
            if count_attacked_vehicle_instances(mem) != 1:
                critical.append(f"S1 attacked instances={count_attacked_vehicle_instances(mem)} (expected 1)")
        elif spec.scenario_id == "S2":
            n = count_attacked_vehicle_instances(mem)
            if n < 2:
                critical.append(f"S2 attacked instances={n} (expected >=2)")
            camps = mem.loc[mem["ground_truth_malicious"] == 1, "ground_truth_campaign_id"].nunique()
            if camps == 1 and n >= 2:
                warnings.append("S2: all attacked vehicles share one campaign_id (check incident IDs)")
        elif spec.scenario_id in ("S3", "S4"):
            n = count_attacked_vehicle_instances(mem)
            if n != 2:
                critical.append(f"{spec.scenario_id} attacked instances={n} (expected 2 for quick test)")
            if "vehicle_token" in mem.columns:
                if mem["vehicle_token"].astype(str).str.contains("Hyundai|Kia|Chevrolet", case=False).any():
                    critical.append(f"{spec.scenario_id}: non-opaque vehicle tokens")

    return critical, warnings


def write_report(critical: list[str], warnings: list[str]) -> Path:
    project_root = resolve_project_root()
    path = project_root / "new_experiments/final_validated_runs/validation/quick_test_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quick Test Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Critical failures:** {len(critical)}",
        f"**Warnings:** {len(warnings)}",
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
    critical, warnings = validate_quick_test()
    report = write_report(critical, warnings)
    print(f"Quick test validation → {report}")
    print(f"Critical: {len(critical)}, Warnings: {len(warnings)}")
    for c in critical:
        print(f"  CRITICAL: {c}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
