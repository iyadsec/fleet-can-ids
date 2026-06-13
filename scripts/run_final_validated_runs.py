#!/usr/bin/env python3
"""Orchestrate final validated experimental pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.result_writer import ProtectedOutputGuard, save_protection_baseline
from src.utils.paths import resolve_project_root

MAIN_CONFIG = "new_experiments/final_validated_runs/configs/final_validated_runs.yaml"
QUICK_CONFIG = "new_experiments/final_validated_runs/configs/final_validated_quick_test.yaml"
CAMPAIGN_CONFIG = "new_experiments/final_validated_runs/configs/final_validated_campaign_analysis.yaml"
OUTPUT_ROOT = "new_experiments/final_validated_runs"
PYTHON = sys.executable


def _run(cmd: list[str], *, cwd: Path) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd)


def freeze_pipeline(cwd: Path) -> int:
    return _run([PYTHON, "scripts/freeze_final_pipeline.py"], cwd=cwd)


def run_quick_test(cwd: Path) -> int:
    guard = ProtectedOutputGuard(cwd, OUTPUT_ROOT)
    guard.ensure_directory_tree()
    save_protection_baseline(guard)
    return _run(
        [
            PYTHON,
            "scripts/run_new_scenario_experiments.py",
            "--config",
            QUICK_CONFIG,
            "--all-scenarios",
            "--quick-test",
            "--methods",
            "local,clustering,standard_gnn,fcgnn",
            "--seeds",
            "11",
            "--campaign-sizes",
            "2",
            "--coordination-strengths",
            "0.0,1.0",
        ],
        cwd=cwd,
    )


def validate_quick_test(cwd: Path) -> int:
    return _run([PYTHON, "scripts/validate_final_quick_test.py"], cwd=cwd)


def run_full_scenarios(cwd: Path) -> int:
    return _run(
        [
            PYTHON,
            "scripts/run_new_scenario_experiments.py",
            "--config",
            MAIN_CONFIG,
            "--all-scenarios",
            "--methods",
            "local,clustering,standard_gnn,fcgnn",
            "--reuse-existing",
        ],
        cwd=cwd,
    )


def run_campaign_analysis(cwd: Path, *, quick: bool = False) -> int:
    cmd = [PYTHON, "scripts/run_campaign_analysis.py", "--config", CAMPAIGN_CONFIG, "--experiment", "all"]
    if quick:
        cmd.append("--quick-test")
    return _run(cmd, cwd=cwd)


def run_edge_sensitivity(cwd: Path) -> int:
    for scenario in ("S3_strong_campaign", "S4_weak_campaign"):
        code = _run(
            [
                PYTHON,
                "scripts/run_new_scenario_experiments.py",
                "--config",
                MAIN_CONFIG,
                "--scenario",
                scenario,
                "--campaign-sizes",
                "5",
                "--coordination-strengths",
                "1.0",
                "--methods",
                "standard_gnn,fcgnn",
                "--edge-sensitivity",
            ],
            cwd=cwd,
        )
        if code != 0:
            return code
    return 0


def build_publication(cwd: Path) -> int:
    return _run(
        [
            PYTHON,
            "scripts/build_final_validated_publication.py",
            "--config",
            MAIN_CONFIG,
        ],
        cwd=cwd,
    )


def validate_final(cwd: Path) -> int:
    return _run([PYTHON, "scripts/validate_final_validated_runs.py"], cwd=cwd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final validated experimental pipeline")
    p.add_argument(
        "--phase",
        choices=[
            "freeze",
            "quick-test",
            "validate-quick",
            "scenarios",
            "campaign",
            "edge",
            "publication",
            "validate",
            "all",
        ],
        default="freeze",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = resolve_project_root()
    phase = args.phase

    if phase in ("freeze", "all"):
        if freeze_pipeline(cwd) != 0:
            return 1

    if phase in ("quick-test", "all"):
        if run_quick_test(cwd) != 0:
            return 1
        if validate_quick_test(cwd) != 0:
            print("Quick test validation failed — stopping.", file=sys.stderr)
            return 1

    if phase == "validate-quick":
        return validate_quick_test(cwd)

    if phase in ("scenarios", "all"):
        if run_full_scenarios(cwd) != 0:
            return 1

    if phase in ("campaign", "all"):
        if run_campaign_analysis(cwd) != 0:
            return 1

    if phase in ("edge", "all"):
        if run_edge_sensitivity(cwd) != 0:
            return 1

    if phase in ("publication", "all"):
        if build_publication(cwd) != 0:
            return 1

    if phase in ("validate", "all"):
        return validate_final(cwd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
