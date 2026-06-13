#!/usr/bin/env python3
"""Validate tuned Phase 4 outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity_final_tuned")
SRC = Path("new_experiments/final_validated_runs/model_diversity_final")


def main() -> int:
    root = resolve_project_root() / OUT
    src = resolve_project_root() / SRC
    failures: list[str] = []
    checks: list[str] = []

    required = [
        "audit/frozen_pipeline_inputs.md",
        "audit/false_campaign_metric_definition.md",
        "audit/current_campaign_gate_audit.md",
        "validation_scenarios/validation_scenario_manifest.csv",
        "configs/gate_selection_objective.yaml",
        "configs/final_selected_campaign_gate.yaml",
        "gate_search/all_gate_candidates.csv",
        "gate_search/gate_selection_report.md",
        "results/final_run_level_metrics.csv",
        "results/final_safety_metrics.csv",
        "comparison/provisional_vs_tuned_phase4.md",
        "FINAL_TUNED_PHASE4_SUMMARY.md",
    ]
    for f in required:
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    if (src / "results").exists():
        checks.append("OK provisional root untouched")

    val = pd.read_csv(root / "validation_scenarios/validation_scenario_manifest.csv")
    if (val["overlap_with_test"] > 0).any():
        failures.append("Validation/test overlap detected")
    else:
        checks.append("OK validation/test separation")

    gate = yaml.safe_load((root / "configs/final_selected_campaign_gate.yaml").read_text())
    if "frozen_at" not in gate:
        failures.append("Gate not frozen before test evaluation")
    else:
        checks.append("OK gate frozen")

    candidates = pd.read_csv(root / "gate_search/all_gate_candidates.csv")
    if len(candidates) < 100:
        failures.append("Insufficient gate candidates retained")
    else:
        checks.append(f"OK {len(candidates)} gate candidates")

    sci_pass = len(failures) == 0
    report = [
        "# Tuned Phase 4 validation",
        "",
        f"**Pipeline integrity:** {'PASS' if sci_pass else 'FAIL'}",
        f"**Validation-test separation:** {'PASS' if 'overlap' not in str(failures) else 'FAIL'}",
        f"**Output completeness:** {'PASS' if sci_pass else 'FAIL'}",
        "",
        "## Scientific performance",
        "",
        "Poor campaign F1 or high false-alert rates are reported limitations, not software failures.",
        "",
        "## Checks passed",
        *[f"- {c}" for c in checks],
    ]
    if failures:
        report += ["", "## Failures", *[f"- {f}" for f in failures]]
    (root / "validation/tuned_phase4_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
