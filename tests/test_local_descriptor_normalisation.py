"""Local benign-training fleet scaler tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.experiments.local_descriptor_normalisation import (
    apply_fleet_scaler,
    fit_benign_fleet_scaler,
)
from src.experiments.metadata_validation import load_split_manifest
from src.experiments.scenario_generator import load_descriptor_tables
from src.graph.fleet_similarity_features import apply_locally_normalized_view
from src.utils.paths import resolve_project_root


class LocalDescriptorNormalisationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = resolve_project_root()
        cls.descriptors, features = load_descriptor_tables(
            root / "data/processed/anomaly_descriptors.csv",
            root / "data/processed/window_features.csv",
        )
        cls.manifest = load_split_manifest(root, cls.descriptors, features)

    def test_scaler_fits_train_benign_only(self) -> None:
        prov = fit_benign_fleet_scaler(self.descriptors, self.manifest)
        self.assertFalse(prov.attack_labels_used)
        self.assertEqual(prov.training_split, "train")
        self.assertGreater(prov.fit_row_count, 0)
        self.assertIn("anomaly_score", prov.fitted_feature_names)

    def test_scaler_does_not_use_vehicle_model_grouping(self) -> None:
        prov = fit_benign_fleet_scaler(self.descriptors, self.manifest)
        global_mean = prov.means.get("anomaly_score")
        self.assertIsNotNone(global_mean)
        # Per-model means would differ; global scaler mean should match benign train global
        join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in self.descriptors.columns]
        desc = self.descriptors.merge(
            self.manifest[join_cols + ["split"]].drop_duplicates(),
            on=join_cols,
            how="left",
        )
        from src.experiments.data_splits import is_benign_attack_type

        train = desc[(desc["split"] == "train") & desc["attack_type"].map(is_benign_attack_type)]
        expected = float(train["anomaly_score"].mean())
        self.assertAlmostEqual(global_mean, expected, places=4)

    def test_test_split_not_used_for_fit(self) -> None:
        prov = fit_benign_fleet_scaler(self.descriptors, self.manifest)
        tampered = self.manifest.copy()
        tampered.loc[tampered["split"] == "test", "split"] = "train"
        prov2 = fit_benign_fleet_scaler(self.descriptors, tampered)
        self.assertNotEqual(prov.fit_row_count, prov2.fit_row_count)

    def test_normalized_view_independent_of_vehicle_model(self) -> None:
        prov = fit_benign_fleet_scaler(self.descriptors, self.manifest)
        sample = self.descriptors.head(20).copy()
        cols = [c for c in prov.fitted_feature_names if c in sample.columns]
        norm = apply_locally_normalized_view(sample, cols, prov)
        hyundai = norm[sample["vehicle_model"] == "Hyundai"]
        kia = norm[sample["vehicle_model"] == "Kia"]
        if len(hyundai) and len(kia) and "anomaly_score" in cols:
            # Same raw value should z-score identically regardless of vehicle_model
            raw = float(sample.loc[sample["vehicle_model"] == "Hyundai", "anomaly_score"].iloc[0])
            z_h = float(norm.loc[sample["vehicle_model"] == "Hyundai", "anomaly_score"].iloc[0])
            fake = sample[sample["vehicle_model"] == "Kia"].head(1).copy()
            fake["anomaly_score"] = raw
            z_k = float(apply_fleet_scaler(fake, prov, feature_names=tuple(cols))["anomaly_score"].iloc[0])
            self.assertAlmostEqual(z_h, z_k, places=5)


if __name__ == "__main__":
    unittest.main()
