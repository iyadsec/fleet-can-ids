#!/usr/bin/env python3
"""Validate campaign analysis experiments and outputs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.campaign_analysis_outputs import collect_run_metrics
from src.experiments.campaign_analysis_writer import CampaignAnalysisGuard, load_campaign_analysis_config
from src.experiments.vehicle_instance_builder import validate_scenario_records
from src.utils.paths import resolve_project_root

DEFAULT_CONFIG = "new_experiments/campaign_analysis/configs/campaign_analysis.yaml"


def validate_runs(output_root: Path, required_seeds: list[int]) -> tuple[list[str], list[str]]:
    critical: list[str] = []
    warnings: list[str] = []

    for experiment in ("campaign_size", "model_diversity"):
        runs_dir = output_root / "results" / experiment / "runs"
        if not runs_dir.exists():
            warnings.append(f"No runs directory for {experiment}")
            continue
        seen_ids: set[str] = set()
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            mapping_path = run_dir / "scenario_vehicle_mapping.csv"
            records_path = run_dir / "selected_source_records.csv"
            metrics_path = run_dir / "run_level_metrics.csv"
            if not metrics_path.exists():
                warnings.append(f"Incomplete run (no metrics): {run_dir.name}")
                continue
            metrics = pd.read_csv(metrics_path).iloc[0]
            if metrics.get("experiment") == "model_diversity":
                run_key = (
                    f"{metrics.get('experiment')}_{metrics.get('attack_strength')}_"
                    f"d{metrics.get('diversity_level')}_{metrics.get('method')}_{metrics.get('seed')}"
                )
            else:
                run_key = (
                    f"{metrics.get('experiment')}_{metrics.get('attack_strength')}_"
                    f"cs{metrics.get('campaign_size')}_{metrics.get('method')}_{metrics.get('seed')}"
                )
            if run_key in seen_ids:
                warnings.append(f"Duplicate condition (multiple runs): {run_key}")
            seen_ids.add(run_key)

            if mapping_path.exists() and records_path.exists():
                mapping = pd.read_csv(mapping_path)
                records = pd.read_csv(records_path)
                cs = int(metrics.get("campaign_size", 0))
                errs = validate_scenario_records(records, mapping, configured_campaign_size=cs)
                critical.extend([f"{run_dir.name}: {e}" for e in errs])

                if experiment == "campaign_size":
                    fleet_sizes = mapping["configured_campaign_size"].unique()
                    if len(set(mapping.groupby("seed").groups)) > 1:
                        pass
                if experiment == "model_diversity":
                    attacked = mapping[mapping["scenario_role"] == "coordinated"]
                    div = attacked["vehicle_model"].nunique()
                    expected_div = int(metrics.get("diversity_level", div))
                    if div != expected_div:
                        critical.append(
                            f"{run_dir.name}: diversity {div} != configured {expected_div}"
                        )

            if records_path.exists():
                records = pd.read_csv(records_path)
                if records["event_id"].duplicated().any():
                    critical.append(f"{run_dir.name}: duplicated event_id")

        df = collect_run_metrics(output_root, experiment)
        if not df.empty:
            present = sorted(df["seed"].unique().tolist())
            missing = [s for s in required_seeds if s not in present]
            if missing and len(present) < len(required_seeds):
                warnings.append(f"{experiment}: missing seeds {missing}")

    return critical, warnings


def validate_tables(output_root: Path) -> list[str]:
    warnings: list[str] = []
    for exp, stems in [
        ("campaign_size", ["table_A1_campaign_size_strong", "table_A2_campaign_size_weak", "table_A3_campaign_size_cost"]),
        ("model_diversity", ["table_B1_model_diversity_strong", "table_B2_model_diversity_weak", "table_B3_cross_model_similarity"]),
    ]:
        for stem in stems:
            csv_p = output_root / "tables" / exp / f"{stem}.csv"
            if not csv_p.exists():
                warnings.append(f"Missing table {csv_p}")
    return warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parsed = parser.parse_args(argv)
    config = load_campaign_analysis_config(parsed.config)
    project_root = resolve_project_root()
    guard = CampaignAnalysisGuard(project_root)
    output_root = guard.output_root
    required_seeds = [int(s) for s in config.get("general", {}).get("seeds", [])]

    critical, warnings = validate_runs(output_root, required_seeds)
    warnings.extend(validate_tables(output_root))

    report_path = output_root / "validation" / "campaign_analysis_validation_report.md"
    lines = [
        "# Campaign Analysis Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Critical failures",
        "",
    ]
    if critical:
        for c in critical:
            lines.append(f"- {c}")
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validation report → {report_path}")
    print(f"Critical: {len(critical)}, Warnings: {len(warnings)}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
