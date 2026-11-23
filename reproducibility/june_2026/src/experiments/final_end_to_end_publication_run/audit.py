"""Audit documents for prior-result exclusion and paper mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

PRIOR_ROOTS: list[tuple[str, str]] = [
    ("results/", "supplementary", "Legacy H1–H4 pipeline CSVs; superseded by this run for Section VII"),
    ("figures/", "supplementary", "Legacy source figures; paper bundle copies retained separately"),
    ("tables/", "supplementary", "Legacy table exports"),
    ("new_experiments/final_validated_runs/", "supplementary", "Phase 2–4 validated grids; not Section VII source"),
    ("new_experiments/final_publication_scenarios/", "superseded", "Prior scenario packages (T/FS/CR tables)"),
    ("new_experiments/archive/", "supplementary", "Archived intermediate outputs"),
    ("data/processed/anomaly_descriptors.csv", "excluded", "Prior global descriptors — not used in this run"),
    ("data/processed/vehicle_anomaly_predictions.csv", "excluded", "Prior global IF scores — not used"),
    ("outputs/models/", "excluded", "Prior pooled IF model — not used"),
]


def write_prior_results_exclusion(out_path: Path) -> None:
    lines = [
        "# Prior results exclusion",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "The final end-to-end publication run is isolated from all prior experiment outputs.",
        "Only raw labelled CAN traces and validated configuration *templates* (copied into this run) are referenced.",
        "",
        "| Root | Status | Notes |",
        "|------|--------|-------|",
    ]
    for root, status, notes in PRIOR_ROOTS:
        lines.append(f"| `{root}` | {status} | {notes} |")
    lines += [
        "",
        "## Allowed inputs",
        "",
        "- Raw OCSLab labelled CAN traces (`configs/fleet_ids.yaml` → `data.external_dataset_dir`)",
        "- Source code modules under `src/`",
        "- Master configuration copied to `configs/final_master_experiment.yaml`",
        "",
        "## Forbidden inputs",
        "",
        "- Any prior run-level metrics, descriptors, models, scalers, or scenario predictions",
        "- Any CSV under archived or superseded experiment roots",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_split_integrity_report(
    trace_df,
    window_manifest,
    output_path: Path,
    *,
    errors: list[str],
) -> None:
    passed = len(errors) == 0
    lines = [
        "# Final split integrity",
        "",
        f"**Validation:** {'PASS' if passed else 'FAIL'}",
        f"**Trace groups:** {len(trace_df)}",
        f"**Windows:** {len(window_manifest)}",
        "",
    ]
    if errors:
        lines += ["## Errors", ""]
        for e in errors:
            lines.append(f"- {e}")
    else:
        lines += [
            "## Checks",
            "",
            "1. Source-trace-level separation — no trace in multiple splits.",
            "2. No overlapping window/event IDs across splits.",
            "3. Benign train/validation/test coverage per OEM.",
            "4. Test windows excluded from IF fitting and threshold selection.",
        ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_paper_artifact_mapping(out_path: Path) -> None:
    sections = [
        ("Experimental Objectives", "table_P1_experimental_objectives", "none", "main"),
        ("Dataset Description", "table_P2_dataset_and_split_summary", "none", "main"),
        ("Experimental Setup", "table_P3_final_experimental_configuration", "none", "main"),
        ("Vehicle-Level Anomaly Detection", "table_P4_vehicle_level_results", "figure_P1_vehicle_level_precision_recall, figure_P2_anomaly_score_distribution", "main"),
        ("Descriptor Compactness and Privacy", "table_P5_descriptor_compactness_and_privacy", "figure_P3_descriptor_bandwidth_scaling", "main"),
        ("Benign, Isolated, and Unrelated Incident Evaluation", "table_P6_benign_isolated_unrelated_results", "none", "main"),
        ("Strong Coordinated Campaign", "table_P7_strong_campaign_results", "figure_P4–P7", "main"),
        ("Weak Coordinated Campaign", "table_P8_weak_campaign_results", "figure_P4–P7", "main"),
        ("Campaign-Size Sensitivity", "table_P9_campaign_size_graph_and_cost", "figure_P5–P7", "main"),
        ("Edge-Connectivity Sensitivity", "table_P10_edge_connectivity_performance, table_P11_edge_connectivity_cost", "figure_P8–P10", "main"),
        ("Statistical Analysis", "table_P12_primary_statistical_tests", "none", "main"),
    ]
    lines = [
        "# Paper artifact mapping (Section VII)",
        "",
        "| Subsection | Table | Figure | Placement |",
        "|------------|-------|--------|-----------|",
    ]
    for title, table, fig, placement in sections:
        lines.append(f"| {title} | {table} | {fig} | {placement} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")
