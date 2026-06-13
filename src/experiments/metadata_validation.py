"""Metadata separation audit utilities (read-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import prepare_gnn_fleet_node_matrix
from src.experiments.experiment_pipeline import build_scenario_graph
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.local_descriptor_normalisation import FleetScalerProvenance
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.scenario_registry import get_scenario
from src.experiments.vehicle_identity import attach_opaque_tokens_for_audit, count_vehicle_model_diversity
from src.graph.fleet_graph_builder import resolve_fleet_similarity_matrix
from src.models.vehicle_ids import _feature_matrix
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS, METADATA_COLUMNS

FORBIDDEN_MODEL_INPUT_NAMES: frozenset[str] = frozenset(
    {
        "vehicle_model",
        "manufacturer",
        "source_vehicle",
        "source_trace",
        "source_file",
        "source_dataset",
        "attack_type",
        "ground_truth_malicious",
        "ground_truth_campaign_id",
        "ground_truth_campaign_member",
        "ground_truth_label",
        "scenario_role",
        "split",
        "campaign_size",
        "coordination_strength",
        "run_id",
        "seed",
        "vin",
        "can_id",
        "scenario_vehicle_id",
        "vehicle_id",
        "vehicle_instance_id",
        "pseudonymous_vehicle_id",
        "vehicle_token",
    }
)

FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "vehicle_model",
    "attack_type",
    "ground_truth",
    "source_file",
    "source_trace",
    "campaign_id",
    "scenario_role",
)

EVALUATION_ONLY_COLUMNS: frozenset[str] = frozenset(
    FORBIDDEN_MODEL_INPUT_NAMES
    | {
        "event_id",
        "window_id",
        "window_key",
        "label",
        "scenario_key",
        "scenario_id",
        "scenario_seed",
        "configured_campaign_size",
        "configured_coordination_strength",
        "evidence_level",
        "local_alert",
        "weak_signal",
        "predicted_malicious",
        "final_decision",
        "cluster_id",
        "method",
    }
)

BEHAVIOURAL_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        "anomaly_score",
        "frame_count",
        "message_rate",
        "burstiness",
        "mean_inter_arrival_time",
        "std_inter_arrival_time",
        "can_id_entropy",
        "most_common_can_id_ratio",
        "payload_entropy",
        "unique_can_id_count",
        "mean_dlc",
        "std_dlc",
        *BEHAVIOURAL_FEATURE_COLUMNS,
    }
)


@dataclass
class MatrixAuditResult:
    pipeline: str
    source_function: str
    source_file: str
    column_names: list[str]
    shape: tuple[int, ...]
    forbidden_hits: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.forbidden_hits


@dataclass
class ValidationFinding:
    severity: str  # critical | warning | info
    check_id: str
    message: str
    location: str = ""


def _check_column_names(columns: Iterable[str], pipeline: str, source: str, path: str) -> MatrixAuditResult:
    cols = list(columns)
    hits = [c for c in cols if c in FORBIDDEN_MODEL_INPUT_NAMES]
    hits += [c for c in cols if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS) and c not in hits]
    warnings: list[str] = []
    if pipeline in {"similarity", "gnn"} and any(c in cols for c in ("can_id_entropy", "most_common_can_id_ratio")):
        warnings.append("CAN-ID aggregate statistics present in behavioural view")
    if pipeline == "isolation_forest" and "unique_can_id_count" in cols:
        warnings.append("unique_can_id_count may encode vehicle/trace identity in local IDS")
    return MatrixAuditResult(
        pipeline=pipeline,
        source_function=source,
        source_file=path,
        column_names=cols,
        shape=(0, len(cols)),
        forbidden_hits=hits,
        warnings=warnings,
    )


def audit_isolation_forest_matrix(descriptors: pd.DataFrame) -> MatrixAuditResult:
    cols = [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in descriptors.columns]
    result = _check_column_names(cols, "isolation_forest", "_feature_matrix", "src/models/vehicle_ids.py")
    result.shape = _feature_matrix(descriptors.head(min(50, len(descriptors)))).shape
    return result


def audit_similarity_matrix(
    descriptors: pd.DataFrame,
    config: dict[str, Any],
    *,
    fleet_scaler_provenance: FleetScalerProvenance,
) -> tuple[MatrixAuditResult, list[str]]:
    graph_cfg = config.get("graph", {})
    _, columns = resolve_fleet_similarity_matrix(
        descriptors,
        similarity_feature_view=graph_cfg.get("similarity_feature_view", "behavior_only_vehicle_normalized"),
        feature_dominance_threshold=5.0,
        allowed_high_dominance_features=frozenset(),
        fleet_scaler_provenance=fleet_scaler_provenance,
    )
    result = _check_column_names(
        columns,
        "cosine_similarity",
        "resolve_fleet_similarity_matrix",
        "src/graph/fleet_similarity_features.py",
    )
    indirect: list[str] = []
    if fleet_scaler_provenance.attack_labels_used:
        indirect.append("Fleet scaler reports attack_labels_used=True (forbidden)")
    if fleet_scaler_provenance.training_split != "train":
        indirect.append(f"Fleet scaler fitted on split={fleet_scaler_provenance.training_split}, expected train")
    return result, indirect


def audit_gnn_node_matrix(
    descriptors: pd.DataFrame,
    *,
    fleet_scaler_provenance: FleetScalerProvenance,
) -> tuple[MatrixAuditResult, list[str]]:
    X, feat_df, cols = prepare_gnn_fleet_node_matrix(
        descriptors, fleet_scaler_provenance=fleet_scaler_provenance
    )
    result = _check_column_names(
        cols, "gnn_data_x", "prepare_gnn_fleet_node_matrix", "src/evaluation/final_gnn_fleet_decision_experiment.py"
    )
    result.shape = X.shape
    indirect: list[str] = []
    if fleet_scaler_provenance.attack_labels_used:
        indirect.append("Fleet scaler reports attack_labels_used=True for GNN node matrix")
    return result, indirect


def audit_graph_construction(
    descriptors: pd.DataFrame,
    config: dict[str, Any],
    seed: int = 42,
) -> tuple[MatrixAuditResult, dict[str, Any], list[str]]:
    gbuild = build_scenario_graph(descriptors, config, seed)
    pyg = gbuild.pyg_data
    cols = [f"dim_{i}" for i in range(int(pyg.x.size(1)))]
    result = _check_column_names(cols, "pyg_data_x", "build_scenario_graph", "src/experiments/experiment_pipeline.py")
    result.shape = tuple(pyg.x.shape)
    side_channels: dict[str, Any] = {
        "x_shape": tuple(pyg.x.shape),
        "edge_index_shape": tuple(pyg.edge_index.shape),
        "has_vehicle_id_tensor": hasattr(pyg, "vehicle_id"),
        "has_attack_id_tensor": hasattr(pyg, "attack_id"),
        "y_unique": int(pyg.y.unique().numel()) if hasattr(pyg, "y") else None,
    }
    notes: list[str] = []
    if hasattr(pyg, "vehicle_id"):
        notes.append(
            "PyG Data.vehicle_id stores integer-coded opaque vehicle instance (not in data.x, not used in GNN forward)"
        )
    if hasattr(pyg, "attack_id"):
        notes.append(
            "PyG Data.attack_id stores integer-coded attack_type (not in data.x, not used in GNN forward)"
        )
    meta = gbuild.meta
    from src.experiments.vehicle_identity import resolve_graph_vehicle_column

    try:
        veh_col = resolve_graph_vehicle_column(meta)
        if meta[veh_col].astype(str).str.match(r"^V_\d{4}$").all():
            notes.append(f"Graph kNN constraints use opaque tokens via `{veh_col}`")
        else:
            notes.append(f"Graph kNN constraints use `{veh_col}` (verify tokens are opaque)")
    except ValueError as exc:
        notes.append(str(exc))
    return result, side_channels, notes


def load_sample_descriptors(project_root: Path, n: int = 200) -> pd.DataFrame:
    desc, _ = load_descriptor_tables(
        project_root / "data/processed/anomaly_descriptors.csv",
        project_root / "data/processed/window_features.csv",
    )
    return desc.sample(n=min(n, len(desc)), random_state=42).reset_index(drop=True)


def load_campaign_analysis_sample(project_root: Path) -> pd.DataFrame | None:
    root = project_root / "new_experiments/campaign_analysis/results/campaign_size/runs"
    if not root.exists():
        return None
    for run_dir in sorted(root.iterdir()):
        p = run_dir / "selected_source_records.csv"
        if p.exists():
            return pd.read_csv(p, nrows=200)
    return None


def classify_column_role(col: str) -> tuple[str, bool, bool, bool]:
    """Return role, used_as_model_input, used_for_evaluation, contains_sensitive_identity."""
    if col in FORBIDDEN_MODEL_INPUT_NAMES:
        return "evaluation_metadata", False, True, col in {"vehicle_model", "source_file", "source_trace", "source_vehicle", "vin", "can_id"}
    if col in BEHAVIOURAL_INPUT_COLUMNS or col in BEHAVIOURAL_FEATURE_COLUMNS:
        return "behavioural_input", True, False, False
    if col in {"anomaly_score", "local_alert", "weak_signal", "evidence_level"}:
        return "ids_output_or_score", True, True, False
    if col in {"event_id", "window_id", "window_key"}:
        return "event_token", False, True, False
    if col.startswith("byte_") or col in {"unique_can_id_count", "mean_dlc", "std_dlc"}:
        return "behavioural_input", True, False, False
    if col in {"predicted_malicious", "final_decision", "cluster_id", "method"}:
        return "prediction_output", False, True, False
    if "ground_truth" in col or col in {"scenario_role", "split", "campaign_size", "coordination_strength"}:
        return "evaluation_metadata", False, True, False
    if col == "scenario_vehicle_id":
        return "pseudonymous_instance_token", False, True, True
    return "other", False, True, False


def inventory_file_columns(file_path: Path, project_root: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        return []
    try:
        if file_path.suffix == ".json":
            return []
        df = pd.read_csv(file_path, nrows=5)
    except Exception:
        return []
    rel = str(file_path.relative_to(project_root))
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        role, model_in, eval_only, sensitive = classify_column_role(str(col))
        rows.append(
            {
                "file_path": rel,
                "column_name": col,
                "dtype": str(df[col].dtype),
                "role": role,
                "used_as_model_input": model_in,
                "used_for_evaluation": eval_only,
                "contains_sensitive_identity": sensitive,
                "notes": "",
            }
        )
    return rows


def discover_inventory_files(project_root: Path) -> list[Path]:
    patterns = [
        "data/processed/anomaly_descriptors.csv",
        "new_experiments/manifests/split_manifest.csv",
        "new_experiments/manifests/results_manifest.csv",
        "new_experiments/campaign_analysis/manifests/available_vehicle_instances.csv",
    ]
    files = [project_root / p for p in patterns if (project_root / p).exists()]

    # One sample per artifact type from scenario + campaign_analysis runs
    sample_globs = [
        "new_experiments/results/S4_weak_campaign/runs/*/scenario_membership.csv",
        "new_experiments/results/S4_weak_campaign/runs/*/selected_source_records.csv",
        "new_experiments/results/S4_weak_campaign/runs/*/event_predictions.csv",
        "new_experiments/results/S4_weak_campaign/runs/*/vehicle_predictions.csv",
        "new_experiments/results/S4_weak_campaign/runs/*/campaign_predictions.csv",
        "new_experiments/results/S4_weak_campaign/runs/*/edge_list.csv",
        "new_experiments/embeddings/S4_weak_campaign/*/node_embeddings.csv",
        "new_experiments/campaign_analysis/results/campaign_size/runs/*/selected_source_records.csv",
        "new_experiments/campaign_analysis/results/campaign_size/runs/*/vehicle_membership.csv",
        "new_experiments/campaign_analysis/results/campaign_size/runs/*/scenario_vehicle_mapping.csv",
        "new_experiments/campaign_analysis/results/campaign_size/runs/*/event_predictions.csv",
    ]
    for pattern in sample_globs:
        matches = sorted(project_root.glob(pattern))
        if matches:
            files.append(matches[0])
    return files


def load_split_manifest(project_root: Path, descriptors: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    manifest_path = project_root / "new_experiments/manifests/split_manifest.csv"
    if manifest_path.exists():
        return pd.read_csv(manifest_path)
    return ensure_split_manifest(
        descriptors,
        features,
        output_path=manifest_path,
        seed=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


def audit_corrected_scenario_identity(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
) -> list[ValidationFinding]:
    """Generate a small S2 scenario and verify opaque instance tokens."""
    from src.experiments.scenario_generator import generate_scenario_records

    findings: list[ValidationFinding] = []
    spec = get_scenario("S2_non_coordinated")
    try:
        scenario_df, membership = generate_scenario_records(
            spec,
            seed=99,
            campaign_size=5,
            coordination_strength=0.0,
            descriptors=descriptors,
            manifest=manifest,
            config=config,
            max_events_per_vehicle=30,
            max_benign_events=50,
        )
    except Exception as exc:
        findings.append(
            ValidationFinding(
                "critical",
                "scenario_generation_failed",
                f"S2 scenario generation failed: {exc}",
                "src/experiments/scenario_generator.py",
            )
        )
        return findings

    for col in ("vehicle_token", "scenario_vehicle_id"):
        if col not in scenario_df.columns:
            findings.append(
                ValidationFinding("critical", f"missing_{col}", f"{col} missing from scenario output", "scenario_generator")
            )
    attacked = membership[membership["ground_truth_malicious"] == 1]
    n_inst = attacked["vehicle_token"].nunique() if "vehicle_token" in attacked.columns else 0
    if n_inst != 5:
        findings.append(
            ValidationFinding(
                "critical",
                "campaign_instance_count",
                f"S2 campaign_size=5 produced {n_inst} attacked instances (expected 5)",
                "scenario_generator",
            )
        )
    if "vehicle_token" in scenario_df.columns:
        non_opaque = scenario_df["vehicle_token"].astype(str).str.contains(
            r"Hyundai|Kia|Chevrolet", case=False, regex=True
        )
        if non_opaque.any():
            findings.append(
                ValidationFinding(
                    "critical",
                    "non_opaque_tokens",
                    "scenario vehicle_token reveals OEM names",
                    "scenario_generator",
                )
            )
    n_models = count_vehicle_model_diversity(membership, malicious_only=True)
    if n_models > n_inst:
        findings.append(
            ValidationFinding(
                "warning",
                "model_diversity_logic",
                "vehicle_model diversity exceeds instance count (unexpected)",
                "vehicle_identity",
            )
        )
    return findings


def run_matrix_audits(project_root: Path) -> tuple[list[MatrixAuditResult], list[ValidationFinding], dict[str, Any]]:
    config = load_experiment_config(project_root / "new_experiments/configs/scenario_experiments.yaml")
    desc_path = project_root / "data/processed/anomaly_descriptors.csv"
    feat_path = project_root / "data/processed/window_features.csv"
    descriptors, features = load_descriptor_tables(desc_path, feat_path)
    manifest = load_split_manifest(project_root, descriptors, features)
    scaler = ensure_fleet_scaler_in_config(config, descriptors, manifest)

    sample = descriptors.sample(n=min(200, len(descriptors)), random_state=42).reset_index(drop=True)
    sample = attach_opaque_tokens_for_audit(sample)
    graph_sample = sample

    results = [
        audit_isolation_forest_matrix(sample),
    ]
    sim_result, sim_indirect = audit_similarity_matrix(sample, config, fleet_scaler_provenance=scaler)
    results.append(sim_result)
    gnn_result, gnn_indirect = audit_gnn_node_matrix(sample, fleet_scaler_provenance=scaler)
    results.append(gnn_result)
    config_with_scaler = {**config, "_fleet_scaler_provenance": scaler}
    graph_result, side_channels, graph_notes = audit_graph_construction(graph_sample, config_with_scaler)
    results.append(graph_result)

    findings: list[ValidationFinding] = []
    for r in results:
        for hit in r.forbidden_hits:
            findings.append(
                ValidationFinding(
                    "critical",
                    f"forbidden_in_{r.pipeline}",
                    f"Forbidden column '{hit}' in {r.pipeline} matrix",
                    r.source_file,
                )
            )
        for w in r.warnings:
            findings.append(ValidationFinding("warning", f"warn_{r.pipeline}", w, r.source_file))

    for note in sim_indirect + gnn_indirect:
        findings.append(
            ValidationFinding(
                "critical",
                "fleet_scaler_provenance",
                note,
                "src/experiments/local_descriptor_normalisation.py",
            )
        )

    for note in graph_notes:
        sev = "critical" if "requires opaque" in note.lower() else "warning"
        findings.append(ValidationFinding(sev, "graph_metadata", note, "src/graph/fleet_graph_builder.py"))

    findings.extend(audit_corrected_scenario_identity(descriptors, manifest, config))

    if scaler.attack_labels_used:
        findings.append(
            ValidationFinding(
                "critical",
                "scaler_used_attack_labels",
                "Fleet scaler attack_labels_used must be False",
                "local_descriptor_normalisation",
            )
        )

    context = {
        "config": config_with_scaler,
        "descriptors": descriptors,
        "manifest": manifest,
        "scaler": scaler,
        "sample": sample,
    }
    return results, findings, context
