#!/usr/bin/env python3
"""Validate final publication scenario outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from src.experiments.final_publication_scenarios.inventory import REQUIRED_SEEDS
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_publication_scenarios")


def main() -> int:
    root = resolve_project_root() / OUT
    failures: list[str] = []
    checks: list[str] = []

    required = [
        "audit/source_results_inventory.csv",
        "audit/source_selection_report.md",
        "results/scenarios/scenario_run_level_metrics.csv",
        "results/scenarios/fleet_campaign_metrics.csv",
        "results/campaign_size/run_level_metrics.csv",
        "validation/publication_artifact_completeness.md",
        "FINAL_SCENARIO_EXPERIMENT_SUMMARY.md",
    ]
    for f in required:
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    inv = pd.read_csv(root / "audit/source_results_inventory.csv")
    if inv[inv["experiment"] == "campaign_size"]["eligible_for_final_publication"].any():
        prelim = inv[(inv["artifact_path"].str.contains("campaign_size/")) & (~inv["artifact_path"].str.contains("corrected"))]
        if prelim["eligible_for_final_publication"].any():
            failures.append("Preliminary campaign_size marked eligible")
        else:
            checks.append("OK preliminary campaign_size excluded")

    cs = pd.read_csv(root / "results/campaign_size/run_level_metrics.csv")
    fcgnn = cs[cs["method"] == "fcgnn"]
    if (fcgnn["graph_nodes"] != 200).any():
        failures.append("Campaign-size node count not 200")
    else:
        checks.append("OK campaign_size 200 nodes")

    fleet = pd.read_csv(root / "results/scenarios/fleet_campaign_metrics.csv")
    if (fleet.get("framework_config", pd.Series()) == "C1").any() and fleet[fleet["framework_config"] == "C1"]["campaign_metrics_na"].eq(False).any():
        failures.append("C1 has non-NA campaign metrics")
    else:
        checks.append("OK C1 campaign metrics N/A")

    edge_path = root / "results/edge_sensitivity/run_level_metrics.csv"
    if edge_path.exists():
        edge = pd.read_csv(edge_path)
        if edge["unique_edges"].isna().all():
            failures.append("Edge sensitivity missing unique_edges")
        else:
            checks.append(f"OK edge sensitivity {len(edge)} runs")

    report = [
        "# Final publication scenarios validation",
        "",
        f"**Result:** {'FAIL' if failures else 'PASS'}",
        "",
        "## Passed",
        *[f"- {c}" for c in checks],
    ]
    if failures:
        report += ["", "## Failures", *[f"- {f}" for f in failures]]
    (root / "validation/final_publication_scenarios_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
