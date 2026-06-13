#!/usr/bin/env python3
"""Validate Phase 3 campaign-size sensitivity outputs."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.campaign_analysis_outputs import collect_run_metrics
from src.experiments.result_writer import ProtectedOutputGuard, verify_protected_directories_unchanged
from src.experiments.vehicle_instance_builder import validate_scenario_records
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
REQUIRED_SIZES = [2, 5, 10]
METHODS = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]
STRENGTHS = ["strong", "weak"]


def _scenario_hash(records: pd.DataFrame) -> str:
    payload = "|".join(sorted(records["event_id"].astype(str)))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_phase3() -> tuple[list[str], list[str]]:
    project_root = resolve_project_root()
    guard = ProtectedOutputGuard(project_root, OUTPUT_ROOT)
    critical: list[str] = []
    warnings: list[str] = []

    ok_legacy, legacy_errs = verify_protected_directories_unchanged(guard)
    if not ok_legacy:
        critical.extend(legacy_errs)

    df = collect_run_metrics(guard.output_root, "campaign_size")
    if df.empty:
        critical.append("No campaign_size run metrics collected")
        return critical, warnings

    if "run_id" in df.columns and df["run_id"].duplicated().any():
        critical.append("Duplicate run_id in aggregated metrics")

    for strength in STRENGTHS:
        for cs in REQUIRED_SIZES:
            for method in METHODS:
                sub = df[
                    (df["attack_strength"] == strength)
                    & (df["campaign_size"] == cs)
                    & (df["method"] == method)
                ]
                seeds = set(sub["seed"].unique()) if not sub.empty else set()
                if len(seeds) < len(REQUIRED_SEEDS):
                    missing = [s for s in REQUIRED_SEEDS if s not in seeds]
                    critical.append(
                        f"Missing runs: {strength} cs={cs} {method} seeds {missing}"
                    )

    runs_dir = guard.output_root / "results" / "campaign_size" / "runs"
    scenario_hashes: dict[tuple, set[str]] = {}
    fleet_sizes: set[int] = set()

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "run_level_metrics.csv"
        if not metrics_path.exists():
            critical.append(f"Incomplete run: {run_dir.name}")
            continue
        m = pd.read_csv(metrics_path).iloc[0]
        cs = int(m.get("campaign_size", 0))
        measured = int(m.get("attacked_vehicle_count", m.get("campaign_size", 0)))
        if measured != cs:
            critical.append(f"{run_dir.name}: attacked instances {measured} != configured {cs}")

        fleet_sizes.add(int(m.get("total_fleet_size", 0)))

        mapping_path = run_dir / "scenario_vehicle_mapping.csv"
        records_path = run_dir / "selected_source_records.csv"
        mapping = pd.read_csv(mapping_path) if mapping_path.exists() else pd.DataFrame()
        records = pd.read_csv(records_path) if records_path.exists() else pd.DataFrame()
        if not mapping.empty and not records.empty:
            errs = validate_scenario_records(records, mapping, configured_campaign_size=cs)
            critical.extend([f"{run_dir.name}: {e}" for e in errs])

        if not records.empty and records["event_id"].duplicated().any():
            critical.append(f"{run_dir.name}: duplicated event_id")

        if not records.empty:
            key = (m.get("attack_strength"), cs, int(m.get("seed", -1)))
            scenario_hashes.setdefault(key, set()).add(_scenario_hash(records))

    for key, hashes in scenario_hashes.items():
        if len(hashes) > 1:
            critical.append(f"Methods differ for condition {key}: {len(hashes)} scenario hashes")

    if len({fs for fs in fleet_sizes if fs > 0}) > 1:
        critical.append(f"Total fleet size not fixed across runs: {fleet_sizes}")

  # Required aggregated outputs
    res = guard.output_root / "results" / "campaign_size"
    stats_path = res / "statistical_tests.csv"
    if stats_path.exists():
        stats_df = pd.read_csv(stats_path)
        if stats_df.empty:
            critical.append("statistical_tests.csv is empty")
        elif len(stats_df) < 18:
            warnings.append(f"statistical_tests.csv has only {len(stats_df)} rows (expected ≥18)")

    for fname in (
        "run_level_metrics.csv",
        "summary_mean_std.csv",
        "statistical_tests.csv",
        "vehicle_membership.csv",
        "scenario_vehicle_mapping.csv",
        "graph_statistics.csv",
        "runtime_memory.csv",
        "confidence_intervals.csv",
    ):
        if not (res / fname).exists():
            critical.append(f"Missing aggregated file: results/campaign_size/{fname}")

    tbl = guard.output_root / "tables"
    for stem in ("table_07a_campaign_size_strong", "table_07b_campaign_size_weak", "table_07c_campaign_size_cost"):
        for ext in (".csv", ".md", ".tex"):
            if not (tbl / f"{stem}{ext}").exists():
                critical.append(f"Missing table {stem}{ext}")

    fig = guard.output_root / "figures"
    for stem in (
        "figure_03_campaign_detection_vs_campaign_size",
        "figure_04_campaign_F1_vs_campaign_size",
        "figure_04b_vehicle_recall_vs_campaign_size",
        "figure_04c_runtime_vs_campaign_size",
    ):
        if not (fig / f"{stem}.pdf").exists():
            critical.append(f"Missing figure {stem}.pdf")
        if not (fig / f"{stem}_weak.pdf").exists():
            critical.append(f"Missing figure {stem}_weak.pdf")

    # Table vs source consistency (row counts)
    src = pd.read_csv(res / "run_level_metrics.csv")
    for stem in ("table_07a_campaign_size_strong", "table_07b_campaign_size_weak"):
        tdf = pd.read_csv(tbl / f"{stem}.csv")
        strength = "strong" if "07a" in stem else "weak"
        expected_rows = len(METHODS) * len(REQUIRED_SIZES)
        if len(tdf) != expected_rows:
            critical.append(f"{stem}: expected {expected_rows} rows, got {len(tdf)}")
        if strength not in src["attack_strength"].values:
            critical.append(f"Source CSV missing attack_strength={strength}")

    return critical, warnings


def main() -> int:
    critical, warnings = validate_phase3()
    path = resolve_project_root() / OUTPUT_ROOT / "validation" / "phase3_campaign_size_validation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 Campaign-Size Validation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Status:** {'PASS' if not critical else 'FAIL'}",
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
    print(f"Phase 3 validation → {path}")
    print(f"Critical: {len(critical)}, Warnings: {len(warnings)}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
