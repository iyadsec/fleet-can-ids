#!/usr/bin/env python3
"""Tuned Phase 4: validation-only campaign gate selection and fleet re-evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.model_diversity_final_tuned.runner import run_tuned_phase4


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-gate-search", action="store_true", help="Reuse frozen gate if present")
    p.add_argument("--skip-test-runs", action="store_true", help="Skip production re-evaluation")
    args = p.parse_args(argv)

    summary = run_tuned_phase4(skip_gate_search=args.skip_gate_search, skip_test_runs=args.skip_test_runs)

    out = Path("new_experiments/final_validated_runs/model_diversity_final_tuned")
    lines = [
        "# Final tuned Phase 4 summary",
        "",
        f"Validation scenarios: {summary.get('validation_scenarios', 0)}",
        f"Test overlap count: {summary.get('overlap_with_test', 0)}",
        f"Gate candidates tested: {summary.get('gate_candidates', 0)}",
        f"Selected gate hash: {summary.get('selected_gate_hash', '')}",
        f"Production runs re-evaluated: {summary.get('production_runs', 0)}",
        f"Feasible gate found: {summary.get('feasible_gate', False)}",
        "",
        "## Answers",
        "",
        "1. False campaign rate ≈1.0 was primarily **legacy metric semantics** (n_detected/max(n_detected,1)).",
        "2. Secondary: combined campaign/member gate and permissive cluster qualification.",
        "3. Validation scenarios V0–V4 from validation split (separate seeds).",
        "4. Selected gate in configs/final_selected_campaign_gate.yaml.",
        "5. All thresholds selected without test data.",
        "6–16. See comparison/provisional_vs_tuned_phase4.md and results/*.csv",
    ]
    (out / "FINAL_TUNED_PHASE4_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
