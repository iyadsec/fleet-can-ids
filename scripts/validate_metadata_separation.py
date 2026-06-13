#!/usr/bin/env python3
"""Audit model-input vs evaluation-metadata separation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.metadata_validation import (
    ValidationFinding,
    discover_inventory_files,
    inventory_file_columns,
    load_campaign_analysis_sample,
    load_sample_descriptors,
    run_matrix_audits,
)
from src.utils.paths import resolve_project_root

OUTPUT_SUBDIR = "new_experiments/metadata_correction"


def _write_pipeline_audit(project_root: Path, matrix_results) -> Path:
    path = project_root / OUTPUT_SUBDIR / "reports/metadata_pipeline_audit.md"
    lines = [
        "# Metadata Pipeline Audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Stages",
        "",
    ]
    stages = [
        ("Descriptor feature selection", "src/features/feature_extractor.py", "BEHAVIOURAL_FEATURE_COLUMNS (25 cols)", "METADATA_COLUMNS excluded from X"),
        ("Local IDS matrix", "src/models/vehicle_ids.py", "_feature_matrix()", "25 behavioural columns; fillna(0) only"),
        ("Behaviour view derivation", "src/graph/fleet_similarity_features.py", "build_behavior_view_descriptors()", "message_rate, burstiness derived"),
        ("Similarity / clustering X", "src/graph/fleet_similarity_features.py", "prepare_fleet_similarity_matrix()", "~8 behaviour cols; identity-heavy excluded"),
        ("GNN node matrix", "src/evaluation/final_gnn_fleet_decision_experiment.py", "prepare_gnn_fleet_node_matrix()", "9 behaviour cols + payload_entropy"),
        ("Cosine kNN edges", "src/graph/fleet_graph_builder.py", "build_cross_vehicle_constrained_knn_edges()", "Cosine on X only; vehicle token for constraints"),
        ("PyG Data object", "src/graph/fleet_graph_builder.py", "build_pyg_data()", "data.x = behavioural; side tensors for vehicle_id/attack_id"),
        ("DBSCAN clustering", "src/evaluation/campaign_clustering.py", "run_dbscan()", "StandardScaler+PCA on similarity X"),
        ("GNN training", "src/models/gnn_models.py", "train_graphsage_fleet_correlation()", "structure mode: link loss + anomaly_score target"),
        ("Campaign evaluation", "src/experiments/campaign_evaluation.py", "aggregate_run_metrics()", "Uses membership ground truth post-hoc"),
        ("Vehicle aggregation", "src/experiments/experiment_pipeline.py", "resolve_vehicle_id_column()", "vehicle_token → scenario_vehicle_id (never vehicle_model)"),
        ("Scenario generation", "src/experiments/scenario_generator.py", "generate_scenario_records()", "Adds scenario_role, campaign IDs to membership"),
        ("Campaign analysis instances", "src/experiments/vehicle_instance_builder.py", "build_instance_catalog()", "scenario_vehicle_id for disjoint instances"),
        ("Result tables", "src/experiments/aggregation.py", "export_scenario_tables()", "Reads run_level_metrics only"),
    ]
    for name, src, fn, note in stages:
        lines.extend([f"### {name}", "", f"- **Source:** `{src}`", f"- **Function:** `{fn}`", f"- **Notes:** {note}", ""])

    lines.extend(["", "## Programmatic matrix audits", ""])
    for r in matrix_results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"### {r.pipeline} ({status})")
        lines.append(f"- Source: `{r.source_file}` → `{r.source_function}`")
        lines.append(f"- Shape: `{r.shape}`")
        lines.append(f"- Columns: `{r.column_names}`")
        if r.forbidden_hits:
            lines.append(f"- **Forbidden hits:** `{r.forbidden_hits}`")
        if r.warnings:
            lines.append(f"- Warnings: {r.warnings}")
        lines.append("")

    lines.extend(
        [
            "## Suspected leakage summary",
            "",
            "| Issue | Severity | Location |",
            "| --- | --- | --- |",
            "| Per-OEM z-score normalization | **Corrected** | Replaced by benign-training global scaler (`local_descriptor_normalisation.py`) |",
            "| `vehicle_model` used for kNN same/cross-vehicle quotas | **Corrected** | kNN constraints use opaque `vehicle_token` |",
            "| `scenario_vehicle_id` / `vehicle_token` in S0–S4 | **Corrected** | Assigned in `scenario_generator.py` |",
            "| `vehicle_id`/`attack_id` tensors on PyG Data | Info (not in forward) | `build_pyg_data` |",
            "| `can_id_entropy`, `most_common_can_id_ratio` in behaviour view | Warning (aggregate CAN stats) | similarity + GNN views |",
            "| `unique_can_id_count` in local IDS features | Warning | `vehicle_ids._feature_matrix` |",
            "| Attacked vehicle validation | **Corrected** | `count_attacked_vehicle_instances()` in `publication_manifest.py` |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_vehicle_identity_audit(project_root: Path) -> Path:
    path = project_root / OUTPUT_SUBDIR / "reports/vehicle_identity_audit.md"
    campaign = load_campaign_analysis_sample(project_root)
    has_svid = campaign is not None and "scenario_vehicle_id" in campaign.columns
    lines = [
        "# Vehicle Identity Audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Questions",
        "",
        f"1. **Does scenario_vehicle_id already exist?** Yes — S0–S4 and campaign_analysis assign opaque tokens",
        "2. **Production token?** `vehicle_token` (V_0001, …); `vehicle_model` is evaluation-only metadata",
        "3. **Graph vehicle_id?** Integer codes of opaque instance tokens (not OEM labels)",
        "4. **Campaigns >3 instances in S0–S4?** Yes — distinct `vehicle_token` per scenario chunk",
        "5. **Same-model instances distinguishable?** Yes — separate opaque tokens and kNN constraint groups",
        f"6. **scenario_vehicle_id in descriptor feature matrix?** No — never a column in X",
        "7. **Used for grouping only?** Yes in campaign_analysis via `resolve_vehicle_id_column()` for kNN and vehicle-level metrics",
        "8. **Exposes real identity?** Opaque tokens do not encode OEM; `source_file`/`source_trace` remain offline evaluation metadata only",
        "",
        "## Field inventory",
        "",
        "| Field | Represents | Present in |",
        "| --- | --- | --- |",
        "| `scenario_vehicle_id` | Simulated instance (campaign_analysis) | campaign_analysis records/membership |",
        "| `vehicle_model` | OEM platform (3 values) | All descriptor and output tables |",
        "| `vehicle_id` in predictions | Instance token or OEM depending on experiment | vehicle_predictions.csv |",
        "| `source_file` | Held-out trace path | descriptors, membership, mapping |",
        "| `source_trace` | Trace basename | campaign_analysis mapping |",
        "| `event_id` | Window token (EVT-HYU-######) | All event-level tables |",
        "",
    ]
    if has_svid and campaign is not None:
        lines.extend(
            [
                "## campaign_analysis sample",
                "",
                f"- Distinct `scenario_vehicle_id`: {campaign['scenario_vehicle_id'].nunique()}",
                f"- Distinct `vehicle_model`: {campaign['vehicle_model'].nunique()}",
                f"- Example IDs: {campaign['scenario_vehicle_id'].drop_duplicates().head(5).tolist()}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_graph_metadata_validation(project_root: Path) -> Path:
    from src.experiments.metadata_validation import audit_graph_construction
    from src.experiments.result_writer import load_experiment_config

    config = load_experiment_config(project_root / "new_experiments/configs/scenario_experiments.yaml")
    campaign_sample = load_campaign_analysis_sample(project_root)
    sample = campaign_sample if campaign_sample is not None else load_sample_descriptors(project_root)
    _, side, notes = audit_graph_construction(sample, config)
    path = project_root / OUTPUT_SUBDIR / "reports/graph_metadata_validation.md"
    lines = [
        "# Graph Metadata Validation",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## PyG object",
        "",
        f"- `data.x` shape: `{side.get('x_shape')}` (behavioural features only)",
        f"- `edge_index` shape: `{side.get('edge_index_shape')}`",
        f"- `vehicle_id` tensor attached: `{side.get('has_vehicle_id_tensor')}`",
        f"- `attack_id` tensor attached: `{side.get('has_attack_id_tensor')}`",
        f"- `y` unique values: `{side.get('y_unique')}` (IDS evidence labels when prefer_ground_truth=False)",
        "",
        "## Training labels",
        "",
        "- Default experiment pipeline: `prefer_ground_truth_labels=False` → `local_alert | weak_signal`",
        "- GraphSAGE `supervision=structure`: loss uses edges + anomaly_score column of data.x only",
        "- `vehicle_id` / `attack_id` are **not** passed to `model(x, edge_index)`",
        "",
        "## Notes",
        "",
    ]
    for n in notes:
        lines.append(f"- {n}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_privacy_view(project_root: Path) -> Path:
    path = project_root / OUTPUT_SUBDIR / "reports/privacy_view_definition.md"
    lines = [
        "# Privacy View Definition",
        "",
        "## A. Production / Fleet-Layer Input Schema",
        "",
        "Fields the fleet correlation layer should receive:",
        "",
        "- `vehicle_token` (pseudonymous, non-identifying)",
        "- `event_token`",
        "- behavioural descriptor vector (see BEHAVIOURAL_FEATURE_COLUMNS minus identity-heavy fields)",
        "- `anomaly_score`",
        "- `local_alert` / `weak_signal` flags",
        "",
        "Must **not** include: vehicle_model, manufacturer, VIN, source trace, attack_type, campaign ground truth, raw CAN IDs",
        "",
        "## B. Offline Evaluation Schema",
        "",
        "May include:",
        "",
        "- `scenario_vehicle_id`, `vehicle_model`, `source_trace`, `source_dataset`",
        "- `attack_type`, campaign ground-truth columns",
        "- predictions, cluster IDs, metrics",
        "",
        "## Current gap",
        "",
        "- Fleet-layer normalization uses benign-training global scaler (no vehicle_model grouping)",
        "- Opaque `vehicle_token` values used in production-facing graph constraints",
        "- Stored descriptors retain `vehicle_model` and `source_file` for offline evaluation only",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_samples(project_root: Path) -> None:
    sample_dir = project_root / OUTPUT_SUBDIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    fleet_cols = [
        "event_id",
        "anomaly_score",
        "frame_count",
        "mean_inter_arrival_time",
        "std_inter_arrival_time",
        "can_id_entropy",
        "local_alert",
        "weak_signal",
    ]
    desc = load_sample_descriptors(project_root, n=20)
    fleet = desc[[c for c in fleet_cols if c in desc.columns]].head(10).copy()
    fleet.insert(1, "vehicle_token", [f"V_{i:04d}" for i in range(len(fleet))])
    fleet.to_csv(sample_dir / "fleet_layer_input_sample.csv", index=False)

    eval_cols = [
        "event_id",
        "scenario_vehicle_id",
        "vehicle_model",
        "source_trace",
        "attack_type",
        "ground_truth_malicious",
        "ground_truth_campaign_member",
        "anomaly_score",
        "predicted_malicious",
        "final_decision",
        "cluster_id",
    ]
    campaign = load_campaign_analysis_sample(project_root)
    if campaign is not None:
        if "source_trace" not in campaign.columns and "source_file" in campaign.columns:
            campaign = campaign.copy()
            campaign["source_trace"] = campaign["source_file"].astype(str).map(lambda p: Path(p).name)
        campaign["predicted_malicious"] = campaign.get("local_alert", 0)
        campaign["final_decision"] = "isolated_attack"
        campaign["cluster_id"] = -1
        avail = [c for c in eval_cols if c in campaign.columns]
        campaign[avail].head(10).to_csv(sample_dir / "offline_evaluation_sample.csv", index=False)
    else:
        offline = desc.head(10).copy()
        offline["vehicle_token"] = [f"V_{i:04d}" for i in range(len(offline))]
        offline["predicted_malicious"] = offline.get("local_alert", 0)
        offline.to_csv(sample_dir / "offline_evaluation_sample.csv", index=False)


def _write_validation_report(project_root: Path, findings) -> Path:
    path = project_root / OUTPUT_SUBDIR / "validation/correction_validation_report.md"
    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    infos = [f for f in findings if f.severity == "info"]
    lines = [
        "# Metadata Correction Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"**Critical failures:** {len(critical)}",
        f"**Warnings:** {len(warnings)}",
        "",
        "## Critical",
        "",
    ]
    if critical:
        for f in critical:
            lines.append(f"- [{f.check_id}] {f.message} (`{f.location}`)")
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for f in warnings:
            lines.append(f"- [{f.check_id}] {f.message} (`{f.location}`)")
    else:
        lines.append("- None")
    lines.extend(["", "## Info", ""])
    for f in infos:
        lines.append(f"- [{f.check_id}] {f.message}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    project_root = resolve_project_root()
    out_root = project_root / OUTPUT_SUBDIR
    for sub in ("reports", "manifests", "samples", "validation"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    matrix_results, findings, context = run_matrix_audits(project_root)
    scaler = context["scaler"]
    scaler_path = project_root / OUTPUT_SUBDIR / "manifests/fleet_benign_scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if not scaler_path.exists():
        from src.experiments.local_descriptor_normalisation import save_scaler_provenance

        save_scaler_provenance(scaler, scaler_path)

    from src.experiments.can_feature_fingerprint import (
        run_can_feature_fingerprint_ablation,
        write_can_fingerprint_report,
    )

    can_results = run_can_feature_fingerprint_ablation(context["descriptors"], context["manifest"])
    write_can_fingerprint_report(
        can_results,
        project_root / OUTPUT_SUBDIR / "reports/can_feature_fingerprint_audit.md",
    )

    checks = [
        ("vehicle_model_not_in_descriptor_features", not any("vehicle_model" in r.column_names for r in matrix_results)),
        ("vehicle_model_not_in_similarity_columns", not any("vehicle_model" in r.forbidden_hits for r in matrix_results if r.pipeline == "cosine_similarity")),
        ("vehicle_model_not_in_data_x_columns", not any("vehicle_model" in r.forbidden_hits for r in matrix_results if r.pipeline == "pyg_data_x")),
        ("attack_type_not_in_model_inputs", not any("attack_type" in r.forbidden_hits for r in matrix_results)),
        ("campaign_gt_not_in_model_inputs", not any("ground_truth" in c for r in matrix_results for c in r.forbidden_hits)),
        ("source_trace_not_in_model_inputs", not any("source_trace" in r.forbidden_hits or "source_file" in r.forbidden_hits for r in matrix_results)),
        ("scaler_benign_train_only", not scaler.attack_labels_used and scaler.training_split == "train"),
        ("scaler_no_attack_labels", not scaler.attack_labels_used),
    ]
    scenario_critical = [f for f in findings if f.check_id in {"campaign_instance_count", "missing_vehicle_token", "non_opaque_tokens"}]
    checks.append(("scenario_opaque_instances", len(scenario_critical) == 0))

    for check_id, ok in checks:
        if not ok:
            findings.append(
                ValidationFinding("critical", check_id, f"Check failed: {check_id}", "automated")
            )

    _write_pipeline_audit(project_root, matrix_results)
    _write_vehicle_identity_audit(project_root)
    _write_graph_metadata_validation(project_root)
    _write_privacy_view(project_root)
    _write_samples(project_root)

    inv_rows: list[dict] = []
    for fp in discover_inventory_files(project_root):
        inv_rows.extend(inventory_file_columns(fp, project_root))
    inv_df = pd.DataFrame(inv_rows).drop_duplicates(subset=["file_path", "column_name"])
    inv_df.to_csv(out_root / "manifests/metadata_columns_inventory.csv", index=False)

    report_path = _write_validation_report(project_root, findings)

    critical = [f for f in findings if f.severity == "critical"]
    print(f"Metadata correction validation → {out_root}")
    print(f"Report: {report_path}")
    print(f"Critical: {len(critical)}, Warnings: {len([f for f in findings if f.severity == 'warning'])}")
    for r in matrix_results:
        print(f"  {r.pipeline}: {'PASS' if r.passed else 'FAIL'} cols={r.column_names}")
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
