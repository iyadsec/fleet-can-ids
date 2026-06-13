#!/usr/bin/env python3
"""Validate corrected Phase 3 campaign-size outputs."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.campaign_analysis_corrected import platform_composition
from src.experiments.campaign_size_corrected_outputs import collect_corrected_metrics, EXPERIMENT
from src.experiments.result_writer import ProtectedOutputGuard, verify_protected_directories_unchanged
from src.experiments.vehicle_instance_builder import validate_scenario_records
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
REQUIRED_SIZES = [2, 5, 10]
METHODS = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]
STRENGTHS = ["strong", "weak"]
EXPECTED_RUNS = 240


def _scenario_hash(records: pd.DataFrame) -> str:
    payload = "|".join(sorted(records["event_id"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_corrected_phase3() -> tuple[list[str], list[str]]:
    project_root = resolve_project_root()
    guard = ProtectedOutputGuard(project_root, OUTPUT_ROOT)
    critical: list[str] = []
    warnings: list[str] = []

    ok_legacy, legacy_errs = verify_protected_directories_unchanged(guard)
    if not ok_legacy:
        critical.extend(legacy_errs)

    orig_metrics = guard.output_root / "results" / "campaign_size" / "run_level_metrics.csv"
    if not orig_metrics.exists():
        warnings.append("Original Phase 3 run_level_metrics.csv not found for comparison")
    else:
        orig_mtime_before = orig_metrics.stat().st_mtime

    df = collect_corrected_metrics(guard.output_root)
    runs_dir = guard.output_root / "results" / EXPERIMENT / "runs"
    n_runs = len(list(runs_dir.glob("*/run_level_metrics.csv"))) if runs_dir.exists() else 0
    if n_runs != EXPECTED_RUNS:
        critical.append(f"Expected {EXPECTED_RUNS} runs, found {n_runs}")

    for strength in STRENGTHS:
        for cs in REQUIRED_SIZES:
            for method in METHODS:
                sub = df[
                    (df["attack_strength"] == strength)
                    & (df["campaign_size"] == cs)
                    & (df["method"] == method)
                ] if not df.empty else pd.DataFrame()
                seeds = set(sub["seed"].unique()) if not sub.empty else set()
                if len(seeds) < len(REQUIRED_SEEDS):
                    missing = [s for s in REQUIRED_SEEDS if s not in seeds]
                    critical.append(f"Missing: {strength} cs={cs} {method} seeds {missing}")

    expected_nodes: set[int] = set()
    fleet_sizes: set[int] = set()
    scenario_hashes: dict[tuple, set[str]] = {}

    for run_dir in sorted(runs_dir.iterdir()) if runs_dir.exists() else []:
        if not run_dir.is_dir():
            continue
        mp = run_dir / "run_level_metrics.csv"
        if not mp.exists():
            critical.append(f"Incomplete run {run_dir.name}")
            continue
        m = pd.read_csv(mp).iloc[0]
        cs = int(m.get("campaign_size", 0))
        strength = str(m.get("attack_strength", ""))
        seed = int(m.get("seed", -1))
        fleet_sizes.add(int(m.get("total_fleet_size", 0)))
        nodes = int(m.get("graph_nodes", len(pd.read_csv(run_dir / "selected_source_records.csv"))))
        expected_nodes.add(nodes)

        records = pd.read_csv(run_dir / "selected_source_records.csv")
        mapping = pd.read_csv(run_dir / "scenario_vehicle_mapping.csv")
        budget_path = run_dir / "event_budget_validation.csv"
        if budget_path.exists():
            bv = pd.read_csv(budget_path).iloc[0]
            if not bool(bv.get("validation_passed", False)):
                critical.append(f"Budget validation failed: {run_dir.name}")

        attacked = records.loc[records["ground_truth_campaign_member"] == 1, "scenario_vehicle_id"].nunique()
        if attacked != cs:
            critical.append(f"{run_dir.name}: attacked instances {attacked} != {cs}")

        vc = records.groupby("scenario_vehicle_id").size()
        if vc.nunique() != 1:
            critical.append(f"{run_dir.name}: unequal descriptors per vehicle")

        if records["event_id"].duplicated().any():
            critical.append(f"{run_dir.name}: duplicate event_id")

        errs = validate_scenario_records(records, mapping, configured_campaign_size=cs)
        critical.extend([f"{run_dir.name}: {e}" for e in errs])

        key = (strength, cs, seed)
        scenario_hashes.setdefault(key, set()).add(_scenario_hash(records))

        if strength == "weak" and (records["vehicle_model"] == "Chevrolet").any():
            mal_chevy = records[
                (records["vehicle_model"] == "Chevrolet")
                & (records["ground_truth_malicious"] == 1)
            ]
            if not mal_chevy.empty:
                critical.append(f"{run_dir.name}: weak campaign contains Chevrolet malicious events")

        comp = platform_composition(strength, cs, seed)  # type: ignore[arg-type]
        if strength == "weak" and comp.get("Chevrolet", 0) > 0:
            critical.append(f"{run_dir.name}: weak composition includes Chevrolet")

    if len(expected_nodes) > 1:
        critical.append(f"Node count not constant across runs: {expected_nodes}")
    if len({f for f in fleet_sizes if f > 0}) > 1:
        critical.append(f"Fleet size not fixed: {fleet_sizes}")

    for key, hashes in scenario_hashes.items():
        if len(hashes) > 1:
            critical.append(f"Methods differ for {key}: {len(hashes)} scenario hashes")

    res = guard.output_root / "results" / EXPERIMENT
    for fname in (
        "run_level_metrics.csv",
        "strong_run_level_metrics.csv",
        "weak_run_level_metrics.csv",
        "statistical_tests_primary.csv",
        "vehicle_composition.csv",
        "event_budget_validation.csv",
        "CORRECTED_PHASE3_SUMMARY.md",
        "original_vs_corrected_phase3.md",
    ):
        if not (res / fname).exists():
            critical.append(f"Missing {fname}")

    tbl = guard.output_root / "tables" / "campaign_size_corrected"
    for stem in (
        "table_C1_strong_campaign_size_results",
        "table_C2_weak_campaign_size_results",
        "table_C3_campaign_size_cost",
        "table_C4_platform_composition",
        "table_C5_primary_statistical_tests",
    ):
        if not (tbl / f"{stem}.csv").exists():
            critical.append(f"Missing table {stem}")

    fig = guard.output_root / "figures" / "campaign_size_corrected"
    for stem in (
        "figure_C1_strong_campaign_detection_vs_size",
        "figure_C2_strong_campaign_F1_vs_size",
        "figure_C3_weak_event_recall_vs_size",
        "figure_C4_weak_campaign_F1_vs_size",
        "figure_C5_vehicle_recall_vs_size",
        "figure_C6_unique_edges_vs_campaign_size",
        "figure_C7_runtime_vs_campaign_size",
    ):
        if not (fig / f"{stem}.pdf").exists():
            critical.append(f"Missing figure {stem}.pdf")

    if orig_metrics.exists() and orig_metrics.stat().st_mtime > orig_mtime_before + 1:
        critical.append("Original Phase 3 run_level_metrics.csv was modified")

    stats = res / "statistical_tests_primary.csv"
    if stats.exists() and pd.read_csv(stats).empty:
        critical.append("statistical_tests_primary.csv is empty")

    return critical, warnings


def main() -> int:
    critical, warnings = validate_corrected_phase3()
    out = (
        resolve_project_root()
        / OUTPUT_ROOT
        / "validation"
        / "campaign_size_corrected"
        / "corrected_phase3_validation.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Corrected Phase 3 Validation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Status:** {'PASS' if not critical else 'FAIL'}",
        "",
        f"**Critical:** {len(critical)}",
        f"**Warnings:** {len(warnings)}",
        "",
        "## Critical",
        "",
    ]
    lines.extend([f"- {c}" for c in critical] if critical else ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- None"])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation → {out} (critical={len(critical)})")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
