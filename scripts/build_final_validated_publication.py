#!/usr/bin/env python3
"""Generate publication tables and figures for final_validated_runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.aggregation import export_scenario_tables, generate_figures, write_final_summary
from src.experiments.result_writer import ProtectedOutputGuard, load_experiment_config
from src.experiments.scenario_registry import SCENARIO_REGISTRY
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=f"{OUTPUT_ROOT}/configs/final_validated_runs.yaml")
    args = parser.parse_args(argv)
    project_root = resolve_project_root()
    guard = ProtectedOutputGuard(project_root, OUTPUT_ROOT)
    config_path = args.config

    for key in SCENARIO_REGISTRY:
        export_scenario_tables(guard.output_root, key)
    generate_figures(guard.output_root)
    write_final_summary(guard.output_root, config_path)

    # Delegate extended publication tables to publication pipeline adapted for this root
    try:
        from src.experiments.final_validated_publication import generate_all_publication_artifacts

        generate_all_publication_artifacts(guard.output_root, load_experiment_config(config_path))
    except ImportError:
        print("Note: final_validated_publication module not yet installed — basic tables/figures only")

    print(f"Publication artifacts → {guard.output_root / 'tables'} and {guard.output_root / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
