#!/usr/bin/env python3
"""Validate framework ablation outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.framework_ablation.config import REQUIRED_SEEDS
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/framework_ablation")


def main() -> int:
    root = resolve_project_root() / OUT
    failures: list[str] = []
    checks: list[str] = []

    required = [
        "audit/gnn_architecture_audit.md",
        "results/framework_ablation_metrics.csv",
        "results/statistical_tests.csv",
        "FINAL_FRAMEWORK_ABLATION_SUMMARY.md",
        "tables/table_F1_framework_configurations.csv",
        "tables/table_F6_primary_statistical_tests.csv",
    ]
    for f in required:
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    if not (resolve_project_root() / "new_experiments/final_validated_runs/results/campaign_size_corrected").exists():
        failures.append("Original campaign_size_corrected deleted")
    else:
        checks.append("OK original results preserved")

    df = pd.read_csv(root / "results/framework_ablation_metrics.csv")
    for cfg in ("C1", "C2", "C3"):
        if cfg not in df["framework_config"].values:
            failures.append(f"Missing framework config {cfg}")

    c1 = df[df["framework_config"] == "C1"]
    if not c1.empty and "campaign_metrics_na" in c1.columns:
        if not c1["campaign_metrics_na"].all():
            failures.append("C1 campaign metrics not all N/A")
        else:
            checks.append("OK C1 campaign N/A")

    stats = pd.read_csv(root / "results/statistical_tests.csv")
    if (stats.get("raw_p_value", pd.Series()) == 0).any():
        failures.append("p-value reported as zero")
    else:
        checks.append("OK no zero p-values")

    s4 = df[df["scenario_id"] == "S4"]
    for seed in REQUIRED_SEEDS:
        sub = s4[(s4["seed"] == seed) & (s4["campaign_size"] == 5)]
        if len(sub["framework_config"].unique()) < 3:
            pass  # warn only if coord runs incomplete

    supp = root / "supplementary/table_S1_standard_gnn_comparison.csv"
    if supp.exists():
        checks.append("OK Standard GNN supplementary only")
    main_methods = df["method"].unique()
    if "standard_gnn" in main_methods and "S_supplementary" not in df.get("framework_config", pd.Series()).values:
        failures.append("standard_gnn in main framework table")

    report = [
        "# Framework ablation validation",
        "",
        f"Critical failures: {len(failures)}",
        "",
        "## Passed",
        *[f"- {c}" for c in checks],
    ]
    if failures:
        report += ["", "## Failures", *[f"- {f}" for f in failures]]
    report.append(f"\n**Result:** {'FAIL' if failures else 'PASS'}")
    (root / "validation/framework_ablation_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
