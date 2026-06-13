"""Campaign instance counting tests."""

from __future__ import annotations

import unittest

from src.experiments.metadata_validation import audit_corrected_scenario_identity, load_split_manifest
from src.experiments.publication_manifest import _validate_scenario_semantics
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import generate_scenario_records, load_descriptor_tables
from src.experiments.scenario_registry import get_scenario
from src.experiments.vehicle_identity import count_attacked_vehicle_instances, count_vehicle_model_diversity
from src.utils.paths import resolve_project_root


class CampaignInstanceCountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = resolve_project_root()
        cls.config = load_experiment_config(root / "new_experiments/configs/scenario_experiments.yaml")
        cls.descriptors, features = load_descriptor_tables(
            root / "data/processed/anomaly_descriptors.csv",
            root / "data/processed/window_features.csv",
        )
        cls.manifest = load_split_manifest(root, cls.descriptors, features)

    def test_s2_campaign_size_counts_instances(self) -> None:
        spec = get_scenario("S2_non_coordinated")
        scenario_df, membership = generate_scenario_records(
            spec,
            seed=42,
            campaign_size=5,
            coordination_strength=0.0,
            descriptors=self.descriptors,
            manifest=self.manifest,
            config=self.config,
            max_events_per_vehicle=25,
            max_benign_events=40,
        )
        n_inst = count_attacked_vehicle_instances(membership)
        self.assertEqual(n_inst, 5)
        n_models = count_vehicle_model_diversity(membership, malicious_only=True)
        self.assertLessEqual(n_models, 3)

    def test_model_diversity_separate_from_instance_count(self) -> None:
        spec = get_scenario("S2_non_coordinated")
        _, membership = generate_scenario_records(
            spec,
            seed=7,
            campaign_size=5,
            coordination_strength=0.0,
            descriptors=self.descriptors,
            manifest=self.manifest,
            config=self.config,
            max_events_per_vehicle=20,
            max_benign_events=30,
        )
        instances = count_attacked_vehicle_instances(membership)
        models = count_vehicle_model_diversity(membership, malicious_only=True)
        self.assertEqual(instances, 5)
        self.assertGreaterEqual(models, 1)

    def test_publication_manifest_uses_instance_count(self) -> None:
        spec = get_scenario("S1_isolated")
        _, membership = generate_scenario_records(
            spec,
            seed=42,
            campaign_size=0,
            coordination_strength=0.0,
            descriptors=self.descriptors,
            manifest=self.manifest,
            config=self.config,
            max_events_per_vehicle=20,
        )
        errors = _validate_scenario_semantics(spec.key, membership, membership, {})
        self.assertFalse(any("attacked vehicles" in e for e in errors if "2" in e))

    def test_audit_corrected_scenario_no_critical(self) -> None:
        findings = audit_corrected_scenario_identity(self.descriptors, self.manifest, self.config)
        critical = [f for f in findings if f.severity == "critical"]
        self.assertEqual(critical, [])


if __name__ == "__main__":
    unittest.main()
