#!/usr/bin/env python3
"""Regenerate paper tables and figures from canonical result bundles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ("verify balanced campaign tables", ["python", "scripts/verify_balanced_campaign_tables.py"]),
    ("baseline/ablation comparison", ["python", "scripts/build_baseline_ablation_comparison.py"]),
    ("consolidate experimental bundle", ["python", "scripts/consolidate_experimental_results.py"]),
    ("P4 strong vs weak figure", ["python", "scripts/build_figure_P4_strong_vs_weak_campaign_f1.py"]),
    ("CORR_EFF3 figure", ["python", "scripts/build_figure_CORR_EFF3_consistency_rule_ablation.py"]),
    ("Overleaf cross-dataset artifacts", ["python", "scripts/build_overleaf_cross_dataset_artifacts.py"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate FLEET-GUARD paper artifacts")
    parser.add_argument("--skip-overleaf", action="store_true", help="Skip Overleaf bundle rebuild")
    args = parser.parse_args()

    failed = 0
    for label, cmd in SCRIPTS:
        if args.skip_overleaf and "Overleaf" in label:
            continue
        print(f"\n=== {label} ===")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED: {label}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
