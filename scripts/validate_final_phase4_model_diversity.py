#!/usr/bin/env python3
"""Validate final Phase 4 outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/model_diversity_final")


def main() -> int:
    root = resolve_project_root() / OUT
    failures, checks = [], []

    required = [
        "manifests/final_split_manifest.csv",
        "manifests/local_model_training_manifest.csv",
        "scalers/scaler_manifest.csv",
        "descriptors/all_descriptors.csv",
        "results/local_ids_by_model.csv",
        "audit/similarity_metric_audit.md",
        "audit/campaign_decision_logic_audit.md",
        "configs/final_campaign_gate.yaml",
    ]
    for f in required:
        if not (root / f).exists():
            failures.append(f"Missing {f}")
        else:
            checks.append(f"OK {f}")

    for archived in ("model_diversity", "model_diversity_corrected"):
        if not (resolve_project_root() / "new_experiments/final_validated_runs" / archived).exists():
            failures.append(f"Archived root missing: {archived}")
        else:
            checks.append(f"OK archived {archived}")

    models = pd.read_csv(root / "manifests/local_model_training_manifest.csv")
    if (models.test_overlap_count != 0).any():
        failures.append("IF training overlaps test")
    else:
        checks.append("OK IF test_overlap_count=0")

    if (root / "results/run_level_metrics.csv").exists():
        run = pd.read_csv(root / "results/run_level_metrics.csv")
        run = run[run.get("is_dry_test", False) != True] if "is_dry_test" in run.columns else run
        if "graph_nodes" in run.columns and not (run.graph_nodes == 200).all():
            failures.append("Node count not 200")
        else:
            checks.append("OK 200 nodes")

    diag = list((root / "results").rglob("similarity_diagnostics.csv"))
    if diag:
        d = pd.concat([pd.read_csv(p) for p in diag])
        if not d.within_valid_range.all():
            failures.append("Cosine similarity out of bounds")
        else:
            checks.append("OK cosine bounds")

    report = ["# Final Phase 4 validation", "", f"**Result:** {'FAIL' if failures else 'PASS'}", "", "## Passed", *[f"- {c}" for c in checks]]
    if failures:
        report += ["", "## Failures", *[f"- {f}" for f in failures]]
    (root / "validation/final_phase4_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
