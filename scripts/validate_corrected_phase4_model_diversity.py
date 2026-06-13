#!/usr/bin/env python3
"""Validate corrected Phase 4 outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity_corrected")
ORIG = Path("new_experiments/final_validated_runs/model_diversity")


def main() -> int:
    root = resolve_project_root() / OUT
    failures = []
    checks = []

    for f in ("audit/raw_dataset_benign_inventory.csv", "manifests/corrected_split_manifest.csv", "results/run_level_metrics.csv"):
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    if (resolve_project_root() / ORIG).exists():
        checks.append("OK original Phase 4 preserved")
    else:
        failures.append("Original Phase 4 removed")

    pool = pd.read_csv(root / "manifests/model_diversity_source_pool_corrected.csv")
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        ben = pool[(pool.vehicle_model == vm) & (pool.available_benign_descriptors > 0)]
        if ben.empty:
            failures.append(f"No benign pool for {vm}")
        else:
            checks.append(f"OK {vm} benign in pool")

    run = pd.read_csv(root / "results/run_level_metrics.csv")
    run = run[run.get("is_dry_test", False) != True] if "is_dry_test" in run.columns else run
    if (run.get("graph_nodes", pd.Series()) != 200).any():
        failures.append("Node count not 200")
    else:
        checks.append("OK 200 nodes")

    if "Hyundai_benign_instances" in run.columns:
        sub = run.groupby("run_id").first()
        if not ((sub.Hyundai_benign_instances == 5) & (sub.Kia_benign_instances == 5) & (sub.Chevrolet_benign_instances == 5)).all():
            failures.append("Benign fleet not 5/5/5")
        else:
            checks.append("OK heterogeneous benign fleet 5/5/5")

    report = ["# Corrected Phase 4 validation", "", f"**Result:** {'FAIL' if failures else 'PASS'}", "", "## Passed", *[f"- {c}" for c in checks]]
    if failures:
        report += ["", "## Failures", *[f"- {f}" for f in failures]]
    (root / "validation/corrected_phase4_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
