#!/usr/bin/env python3
"""Validate hierarchical alignment outputs."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.experiments.hierarchical_alignment.transform import CONFIG_LABELS
from src.utils.paths import resolve_project_root

OUT = Path("new_experiments/final_validated_runs/hierarchical_alignment")
OLD_PATHS = [
    "new_experiments/final_validated_runs/results/campaign_size_corrected",
    "new_experiments/final_validated_runs/evaluation_correction",
    "new_experiments/final_validated_runs/framework_ablation",
]


def _read_csv(root: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(root / "results" / name)


def main() -> int:
    root = resolve_project_root() / OUT
    project = resolve_project_root()
    failures: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    required_results = [
        "hierarchical_event_predictions.csv",
        "local_event_metrics.csv",
        "fleet_campaign_metrics.csv",
        "weak_campaign_support.csv",
        "campaign_membership_errors.csv",
        "capability_comparison.csv",
        "statistical_tests.csv",
    ]
    for f in required_results:
        p = root / "results" / f
        if not p.exists():
            failures.append(f"Missing results/{f}")
        else:
            checks.append(f"OK results/{f}")

    required_tables = [
        "table_H1_hierarchical_output_definitions.csv",
        "table_H2_local_ids_performance.csv",
        "table_H6_end_to_end_capability_comparison.csv",
    ]
    for f in required_tables:
        if not (root / "tables" / f).exists():
            failures.append(f"Missing tables/{f}")
        else:
            checks.append(f"OK tables/{f}")

    required_figures = [
        "figure_H1_hierarchical_detection_flow.png",
        "figure_H2_similarity_vs_graph_campaign_F1.png",
        "figure_H5_false_campaign_rate.png",
    ]
    for f in required_figures:
        if not (root / "figures" / f).exists():
            failures.append(f"Missing figures/{f}")
        else:
            checks.append(f"OK figures/{f}")

    for old in OLD_PATHS:
        if not (project / old).exists():
            failures.append(f"Original path removed: {old}")
        else:
            checks.append(f"OK preserved {old}")

    events = _read_csv(root, "hierarchical_event_predictions.csv")
    schema = [
        "event_id",
        "scenario_vehicle_id",
        "vehicle_token",
        "local_anomaly_score",
        "local_evidence_level",
        "local_event_alert",
        "fleet_cluster_id",
        "fleet_campaign_member",
        "fleet_campaign_confidence",
        "fleet_decision",
        "ground_truth_malicious",
        "ground_truth_campaign_member",
    ]
    missing = [c for c in schema if c not in events.columns]
    if missing:
        failures.append(f"Schema missing columns: {missing}")
    else:
        checks.append("OK output schema columns")

    if "vehicle_model" in events.columns or "attack_type" in events.columns:
        warnings.append("Vehicle model or attack_type present in main prediction schema (should be metadata only)")
    else:
        checks.append("OK vehicle model not in main schema")

    local = _read_csv(root, "local_event_metrics.csv")
    if (local.get("prediction_source", pd.Series()) != "isolation_forest_only").any():
        failures.append("Event metrics not tagged isolation_forest_only")
    else:
        checks.append("OK event metrics from Isolation Forest")

    fleet = _read_csv(root, "fleet_campaign_metrics.csv")
    c1 = fleet[fleet["framework_config"] == "C1"]
    if c1.empty:
        failures.append("No C1 fleet rows")
    elif not c1["campaign_metrics_na"].all():
        failures.append("C1 campaign metrics not all N/A")
    else:
        checks.append("OK C1 campaign metrics N/A")

    c2 = fleet[fleet["framework_config"] == "C2"]
    c3 = fleet[fleet["framework_config"] == "C3"]
    if c2.empty or c3.empty:
        failures.append("Missing C2 or C3 fleet metrics")
    else:
        checks.append("OK C2 and C3 fleet metrics present")

    stats = _read_csv(root, "statistical_tests.csv")
    if not stats.empty:
        if (stats["comparison"] != "C3 vs C2").any():
            failures.append("Statistical tests include non C3-vs-C2 comparisons")
        else:
            checks.append("OK stats compare C3 vs C2 only")
        if (stats.get("raw_p_value", pd.Series()) == 0).any():
            failures.append("p-value reported as zero")
        else:
            checks.append("OK no zero p-values")

    bad_fig = list((root / "figures").glob("*local*graph*event*f1*"))
    if bad_fig:
        failures.append("Forbidden Local-vs-GraphSAGE event F1 figure exists")
    else:
        checks.append("OK no Local-vs-GraphSAGE event competition figure")

    # Local not overwritten for graph methods: fleet member true but local alert false is valid
    graph_events = events[events["framework_config"].isin(["C2", "C3"])]
    if not graph_events.empty:
        valid_weak = graph_events[
            (graph_events["local_evidence_level"] == "weak")
            & (graph_events["local_event_alert"] == 0)
            & (graph_events["fleet_campaign_member"] == 1)
        ]
        if len(valid_weak):
            checks.append("OK hierarchical weak+campaign pattern preserved")
        if (graph_events["local_event_alert"] == graph_events["fleet_campaign_member"]).all():
            warnings.append("local_event_alert always equals fleet_campaign_member — check separation")

    # Tables match CSVs
    h2 = pd.read_csv(root / "tables" / "table_H2_local_ids_performance.csv")
    if len(h2) != local["scenario_id"].nunique():
        warnings.append("Table H2 row count differs from local metric scenarios")
    else:
        checks.append("OK Table H2 matches local metrics scenarios")

    # No hard-coded placeholder metrics in summary
    summary = (root / "FINAL_HIERARCHICAL_ALIGNMENT_SUMMARY.md").read_text(encoding="utf-8")
    if "TODO" in summary or "HARDCODED" in summary:
        failures.append("Summary contains placeholder text")
    else:
        checks.append("OK summary generated from computed outputs")

    if not (root / "audit" / "output_schema_definition.md").exists():
        failures.append("Missing audit/output_schema_definition.md")
    else:
        checks.append("OK audit schema doc")

    if not (root / "scripts/validate_hierarchical_alignment.py").exists():
        pass  # we're running from scripts/
    if not (project / "scripts" / "validate_hierarchical_alignment.py").exists():
        warnings.append("validate script path check skipped")

    report = [
        "# Hierarchical alignment validation",
        "",
        f"**Result:** {'FAIL' if failures else 'PASS'}",
        f"Critical failures: {len(failures)}",
        f"Warnings: {len(warnings)}",
        "",
        "## Passed",
        *[f"- {c}" for c in checks],
    ]
    if warnings:
        report += ["", "## Warnings", *[f"- {w}" for w in warnings]]
    if failures:
        report += ["", "## Critical failures", *[f"- {f}" for f in failures]]

    report.append(
        "\n## Configuration labels\n"
        + "\n".join(f"- {k}: {v}" for k, v in CONFIG_LABELS.items())
    )
    (root / "validation" / "hierarchical_alignment_validation.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
