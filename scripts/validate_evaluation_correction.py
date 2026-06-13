#!/usr/bin/env python3
"""Validate evaluation correction outputs and decision logic."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.evaluation_correction.promotion import PromotionConfig, apply_corrected_event_decisions
from src.experiments.evaluation_correction.statistics import format_p_value
from src.utils.paths import resolve_project_root

OUT_ROOT = Path("new_experiments/final_validated_runs/evaluation_correction")
PHASE3_ROOT = Path("new_experiments/final_validated_runs/results/campaign_size_corrected")


def main() -> int:
    project_root = resolve_project_root()
    out = project_root / OUT_ROOT
    failures: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    required_dirs = ["audit", "results", "tables", "figures", "validation", "fixed_evidence_control"]
    for d in required_dirs:
        if not (out / d).exists():
            failures.append(f"Missing directory: {d}")

    required_files = [
        "audit/event_prediction_logic_audit.md",
        "audit/fpr_root_cause_analysis.md",
        "audit/final_decision_schema.md",
        "audit/campaign_f1_decline_analysis.md",
        "results/event_confusion_counts.csv",
        "results/vehicle_level_detailed_metrics.csv",
        "results/campaign_error_breakdown.csv",
        "results/statistical_tests_corrected.csv",
        "results/benign_on_attacked_audit.csv",
        "FINAL_EVALUATION_CORRECTION_SUMMARY.md",
    ]
    for f in required_files:
        if not (out / f).exists():
            failures.append(f"Missing file: {f}")
        else:
            checks.append(f"OK file {f}")

    # Original outputs not overwritten
    if (project_root / PHASE3_ROOT / "run_level_metrics.csv").exists():
        checks.append("OK original Phase 3 metrics preserved")
    corr_pred = out / "results" / "corrected_predictions"
    if corr_pred.exists() and any(corr_pred.iterdir()):
        checks.append("OK corrected predictions separate from Phase 3")
    if not (project_root / PHASE3_ROOT / "runs").exists():
        failures.append("Original Phase 3 runs missing")

    conf = pd.read_csv(out / "results" / "event_confusion_counts.csv")
    if conf.empty:
        failures.append("event_confusion_counts empty")
    else:
        if not ((conf["tp"] + conf["fp"] + conf["tn"] + conf["fn"]) > 0).all():
            failures.append("Invalid confusion counts")
        if (conf["predicted_benign_events"] > 0).any():
            checks.append("OK event predictions have negative path")
        else:
            failures.append("No negative event predictions in corrected evaluation")
        if (conf["fpr"] < 1.0).any():
            checks.append("OK FPR below 1.0 for some runs")
        else:
            warnings.append("All corrected FPR still 1.0")

    stats = pd.read_csv(out / "results" / "statistical_tests_corrected.csv")
    if not stats.empty:
        if (stats["raw_p_value"] == 0).any():
            failures.append("Raw p-value reported as exactly 0")
        else:
            checks.append("OK no zero raw p-values")
        if "adjusted_p_value_formatted" in stats.columns:
            if stats["adjusted_p_value_formatted"].astype(str).str.fullmatch("0(\\.0+)?").any():
                failures.append("Formatted p-value is zero")

    m1 = pd.read_csv(out / "results" / "corrected_run_level_metrics.csv")
    m1_rows = m1[m1["method"] == "local_ids"]
    if not m1_rows.empty and "campaign_metrics_na" in m1_rows.columns:
        if m1_rows["campaign_metrics_na"].all():
            checks.append("OK M1 campaign metrics N/A")
        else:
            failures.append("M1 campaign metrics not marked N/A")

    benign = pd.read_csv(out / "results" / "benign_on_attacked_audit.csv")
    if not benign.empty:
        bad = benign[benign["window_ground_truth_malicious"] == 1]
        if len(bad):
            failures.append(f"{len(bad)} malicious windows relabelled benign in audit")
        else:
            checks.append("OK no GT malicious relabelled benign")

    fixed_path = out / "fixed_evidence_control" / "run_level_metrics.csv"
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path)
        if "error" not in fixed.columns or fixed["error"].isna().all():
            checks.append("OK fixed-evidence runs completed")
        else:
            warnings.append(f"Fixed-evidence errors: {fixed['error'].dropna().tolist()[:3]}")

    # Spot-check promotion logic
    sample_run = next((project_root / PHASE3_ROOT / "runs").iterdir())
    if (sample_run / "event_predictions.csv").exists():
        raw = pd.read_csv(sample_run / "event_predictions.csv")
        m = pd.read_csv(sample_run / "run_level_metrics.csv").iloc[0]
        corrected = apply_corrected_event_decisions(
            raw,
            attack_strength=str(m["attack_strength"]),
            method=str(m["method"]),
            cfg=PromotionConfig(),
        )
        coord_only = (
            (corrected.get("predicted_campaign_membership", 0) == 1)
            & (corrected["predicted_malicious"] == 0)
        ).sum()
        if coord_only > 0:
            checks.append("OK campaign membership does not force all events malicious")
        ben_promo = int(
            (
                (corrected["ground_truth_malicious"] == 0)
                & (corrected["predicted_malicious"] == 1)
                & (corrected.get("weak_malicious_promoted", 0) == 0)
                & (corrected["local_evidence_level"] == "benign")
            ).sum()
        )
        if ben_promo == len(corrected):
            warnings.append("Many benign GT promoted without weak promotion flag")

    report_lines = [
        "# Evaluation correction validation report",
        "",
        f"**Critical failures:** {len(failures)}",
        f"**Warnings:** {len(warnings)}",
        "",
        "## Checks passed",
        *[f"- {c}" for c in checks],
        "",
    ]
    if failures:
        report_lines += ["## Critical failures", *[f"- {f}" for f in failures], ""]
    if warnings:
        report_lines += ["## Warnings", *[f"- {w}" for w in warnings], ""]

    report_lines.append(f"\n**Result:** {'FAIL' if failures else 'PASS'}")
    (out / "validation" / "evaluation_correction_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
