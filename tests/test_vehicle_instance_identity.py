"""Opaque vehicle-instance identity tests."""

from __future__ import annotations

import re
import unittest

from src.experiments.metadata_validation import audit_corrected_scenario_identity, load_split_manifest
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import load_descriptor_tables
from src.experiments.vehicle_identity import (
    VehicleTokenAllocator,
    assign_identity_to_chunk,
    resolve_graph_vehicle_column,
)
from src.graph.fleet_graph_builder import build_pyg_data
from src.utils.paths import resolve_project_root


class VehicleInstanceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = resolve_project_root()
        cls.config = load_experiment_config(root / "new_experiments/configs/scenario_experiments.yaml")
        desc, feat = load_descriptor_tables(
            root / "data/processed/anomaly_descriptors.csv",
            root / "data/processed/window_features.csv",
        )
        cls.manifest = load_split_manifest(root, desc, feat)
        cls.descriptors = desc

    def test_opaque_token_format(self) -> None:
        alloc = VehicleTokenAllocator()
        chunk = self.descriptors.head(5).copy()
        out = assign_identity_to_chunk(
            chunk,
            allocator=alloc,
            instance_key="t1",
            vehicle_model="Hyundai",
            source_file="trace.csv",
        )
        self.assertTrue(bool(re.match(r"^V_\d{4}$", out["vehicle_token"].iloc[0])))
        self.assertEqual(out["vehicle_token"].iloc[0], out["scenario_vehicle_id"].iloc[0])

    def test_resolve_graph_vehicle_column_prefers_token(self) -> None:
        df = self.descriptors.head(3).copy()
        df["vehicle_token"] = ["V_0001", "V_0002", "V_0003"]
        df["scenario_vehicle_id"] = ["S1", "S2", "S3"]
        self.assertEqual(resolve_graph_vehicle_column(df), "vehicle_token")

    def test_same_model_distinct_instances(self) -> None:
        findings = audit_corrected_scenario_identity(self.descriptors, self.manifest, self.config)
        critical = [f for f in findings if f.severity == "critical"]
        self.assertEqual(critical, [], msg=str(critical))

    def test_pyg_vehicle_id_not_in_feature_matrix(self) -> None:
        import numpy as np

        alloc = VehicleTokenAllocator()
        chunk = assign_identity_to_chunk(
            self.descriptors.head(10).copy(),
            allocator=alloc,
            instance_key="pyg-test",
            vehicle_model="Kia",
            source_file="x.csv",
        )
        x = chunk[["anomaly_score", "frame_count"]].fillna(0).to_numpy(dtype=np.float32)
        ei = np.zeros((2, 0), dtype=np.int64)
        ew = np.zeros(0, dtype=np.float32)
        data = build_pyg_data(x, ei, ew, chunk, prefer_ground_truth_labels=False)
        self.assertEqual(data.x.shape[1], 2)
        self.assertTrue(hasattr(data, "vehicle_id"))


if __name__ == "__main__":
    unittest.main()
