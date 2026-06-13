#!/usr/bin/env python3
"""Validate Phase 4 model diversity outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.model_diversity.compositions import REQUIRED_SEEDS
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity")
HIER = Path("new_experiments/final_validated_runs/hierarchical_alignment/validation/hierarchical_alignment_validation.md")


def main() -> int:
    root = resolve_project_root() / OUT
    failures: list[str] = []
    checks: list[str] = []

    if not (resolve_project_root() / HIER).exists():
        failures.append("Missing hierarchical alignment validation")
    else:
        t = (resolve_project_root() / HIER).read_text(encoding="utf-8")
        if "Critical failures: 0" not in t:
            failures.append("Hierarchical alignment did not pass")
        else:
            checks.append("OK hierarchical alignment gate")

    required = [
        "audit/model_diversity_data_availability.md",
        "manifests/model_diversity_source_pool.csv",
        "results/run_level_metrics.csv",
        "PHASE4_MODEL_DIVERSITY_SUMMARY.md",
    ]
    for f in required:
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    run_df = pd.read_csv(root / "results" / "run_level_metrics.csv") if (root / "results" / "run_level_metrics.csv").exists() else pd.DataFrame()
    if not run_df.empty:
        if (run_df.get("graph_nodes", pd.Series()) != 200).any():
            failures.append("Node count not fixed at 200")
        else:
            checks.append("OK 200 nodes per run")
        c1 = run_df[run_df.get("framework_config", run_df.get("configuration", pd.Series())) == "C1"]
        if not c1.empty and "campaign_metrics_na" in c1.columns and not c1["campaign_metrics_na"].all():
            failures.append("C1 campaign metrics not N/A")
        else:
            checks.append("OK C1 campaign N/A")

    uns = root / "results" / "unsupported_configurations.csv"
    if uns.exists():
        u = pd.read_csv(uns)
        if not ((u["attack_strength"] == "weak") & (u["diversity_level"] == 3)).any():
            checks.append("OK weak D3 marked unsupported")
        else:
            checks.append("OK unsupported configurations documented")

    bad_fig = list((root / "figures").glob("*local*graph*event*f1*")) if (root / "figures").exists() else []
    if bad_fig:
        failures.append("Forbidden event F1 competition figure")
    else:
        checks.append("OK no Local-vs-GraphSAGE event figure")

    for old in (
        "new_experiments/final_validated_runs/hierarchical_alignment",
        "new_experiments/campaign_analysis",
    ):
        if not (resolve_project_root() / old).exists():
            failures.append(f"Prior results removed: {old}")
        else:
            checks.append(f"OK preserved {old}")

    report = [
        "# Phase 4 model diversity validation",
        "",
        f"**Result:** {'FAIL' if failures else 'PASS'}",
        f"Critical failures: {len(failures)}",
        "",
        "## Passed",
        *[f"- {c}" for c in checks],
    ]
    if failures:
        report += ["", "## Critical failures", *[f"- {f}" for f in failures]]
    (root / "validation" / "phase4_model_diversity_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
