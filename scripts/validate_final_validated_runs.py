#!/usr/bin/env python3
"""Final validation for new_experiments/final_validated_runs/."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.publication_manifest import REQUIRED_SEEDS, scan_runs
from src.experiments.result_writer import ProtectedOutputGuard, verify_protected_directories_unchanged
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"
REQUIRED_TABLES = [f"table_{i:02d}" for i in range(1, 13)]


def validate() -> tuple[list[str], list[str]]:
    project_root = resolve_project_root()
    guard = ProtectedOutputGuard(project_root, OUTPUT_ROOT)
    critical: list[str] = []
    warnings: list[str] = []

    quick_report = guard.output_root / "validation/quick_test_report.md"
    if not quick_report.exists():
        critical.append("quick_test_report.md missing — run Phase 1 first")
    elif "Critical failures:** 0" not in quick_report.read_text(encoding="utf-8"):
        if "**Critical failures:** 0" not in quick_report.read_text(encoding="utf-8"):
            critical.append("Quick test did not pass (see quick_test_report.md)")

    ok, errs = verify_protected_directories_unchanged(guard)
    if not ok:
        critical.extend(errs)

    snapshot = guard.output_root / "provenance/pipeline_snapshot.md"
    if not snapshot.exists():
        critical.append("pipeline_snapshot.md missing")

    for scenario in ("S0_benign_control", "S1_isolated", "S2_non_coordinated", "S3_strong_campaign", "S4_weak_campaign"):
        runs = scan_runs(guard.output_root / "results" / scenario)
        seeds = {r.seed for r in runs if not r.excluded}
        if len(seeds) < len(REQUIRED_SEEDS):
            warnings.append(f"{scenario}: only {len(seeds)}/{len(REQUIRED_SEEDS)} seeds completed")

    tables_dir = guard.output_root / "tables"
    for prefix in REQUIRED_TABLES:
        if not list(tables_dir.glob(f"{prefix}*")):
            warnings.append(f"Missing table artifact: {prefix}")

    manifest = guard.output_root / "manifests/results_manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        if df["run_id"].duplicated().any():
            critical.append("Duplicate run_id in results_manifest.csv")

    return critical, warnings


def main() -> int:
    critical, warnings = validate()
    path = resolve_project_root() / OUTPUT_ROOT / "validation/final_validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Validated Runs — Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
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
    print(f"Final validation → {path}")
    print(f"Critical: {len(critical)}, Warnings: {len(warnings)}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
