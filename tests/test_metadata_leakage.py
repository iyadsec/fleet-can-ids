"""Automated metadata leakage tests — inspect final model-input matrices."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.metadata_validation import (
    FORBIDDEN_MODEL_INPUT_NAMES,
    audit_gnn_node_matrix,
    audit_graph_construction,
    audit_isolation_forest_matrix,
    audit_similarity_matrix,
    load_sample_descriptors,
    load_split_manifest,
)
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import load_descriptor_tables
from src.experiments.vehicle_identity import attach_opaque_tokens_for_audit
from src.utils.paths import resolve_project_root


def _assert_no_forbidden(columns: list[str], pipeline: str) -> None:
    hits = [c for c in columns if c in FORBIDDEN_MODEL_INPUT_NAMES]
    if hits:
        raise AssertionError(f"{pipeline}: forbidden columns in model input: {hits}")


class MetadataLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = resolve_project_root()
        cls.sample = attach_opaque_tokens_for_audit(
            load_sample_descriptors(cls.project_root, n=150)
        )
        cls.config = load_experiment_config(
            cls.project_root / "new_experiments/configs/scenario_experiments.yaml"
        )
        desc, feat = load_descriptor_tables(
            cls.project_root / "data/processed/anomaly_descriptors.csv",
            cls.project_root / "data/processed/window_features.csv",
        )
        manifest = load_split_manifest(cls.project_root, desc, feat)
        cls.scaler = ensure_fleet_scaler_in_config(cls.config, desc, manifest)

    def test_isolation_forest_matrix_no_metadata(self) -> None:
        result = audit_isolation_forest_matrix(self.sample)
        _assert_no_forbidden(result.column_names, result.pipeline)

    def test_similarity_matrix_no_metadata(self) -> None:
        result, indirect = audit_similarity_matrix(
            self.sample, self.config, fleet_scaler_provenance=self.scaler
        )
        _assert_no_forbidden(result.column_names, result.pipeline)
        self.assertEqual(indirect, [])

    def test_gnn_node_matrix_no_metadata(self) -> None:
        result, indirect = audit_gnn_node_matrix(
            self.sample, fleet_scaler_provenance=self.scaler
        )
        _assert_no_forbidden(result.column_names, result.pipeline)
        self.assertEqual(indirect, [])

    def test_scaler_benign_train_only(self) -> None:
        self.assertFalse(self.scaler.attack_labels_used)
        self.assertEqual(self.scaler.training_split, "train")

    def test_pyg_data_x_no_metadata(self) -> None:
        cfg = {**self.config, "_fleet_scaler_provenance": self.scaler}
        result, side, _ = audit_graph_construction(self.sample, cfg)
        _assert_no_forbidden(result.column_names, result.pipeline)
        self.assertEqual(side["x_shape"][1], len(result.column_names))

    def test_gnn_side_channels_not_feature_dims(self) -> None:
        cfg = {**self.config, "_fleet_scaler_provenance": self.scaler}
        _, side, _ = audit_graph_construction(self.sample, cfg)
        self.assertTrue(side["has_vehicle_id_tensor"])
        self.assertTrue(side["has_attack_id_tensor"])
        result, _, _ = audit_graph_construction(self.sample, cfg)
        self.assertLess(result.shape[1], 20)


if __name__ == "__main__":
    unittest.main()
