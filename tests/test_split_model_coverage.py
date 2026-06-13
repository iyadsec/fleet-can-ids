"""Corrected split must cover all models in test benign."""

import pandas as pd

from src.experiments.data_splits import build_split_manifest_balanced_benign, validate_model_benign_test_coverage, is_benign_attack_type


def test_balanced_benign_split_covers_all_models() -> None:
    rows = []
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        for atk in ("attack_free", "malfunction"):
            for i in range(20):
                rows.append({"window_id": i, "vehicle_model": vm, "source_file": f"{vm}_{atk}_{i//10}.csv", "attack_type": atk})
    meta = pd.DataFrame(rows)
    manifest = build_split_manifest_balanced_benign(meta, seed=42)
    assert not validate_model_benign_test_coverage(manifest)
    test_ben = manifest[(manifest.split == "test") & manifest.attack_type.map(is_benign_attack_type)]
    assert set(test_ben.vehicle_model.unique()) == {"Hyundai", "Kia", "Chevrolet"}
