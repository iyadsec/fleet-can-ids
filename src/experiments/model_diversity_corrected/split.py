"""Corrected split manifest for model-diversity experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.data_splits import (
    build_split_manifest_balanced_benign,
    is_benign_attack_type,
    validate_model_benign_test_coverage,
    validate_no_split_leakage,
)
from src.experiments.scenario_generator import load_descriptor_tables


def build_corrected_split_manifest(
    descriptors_path: Path,
    features_path: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    descriptors, features = load_descriptor_tables(descriptors_path, features_path)
    meta = features.drop_duplicates(subset=["window_id", "vehicle_model", "source_file"])
    manifest = build_split_manifest_balanced_benign(
        meta,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        min_benign_test_files_per_model=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    errors = validate_no_split_leakage(manifest, ("source_file", "vehicle_model"))
    errors.extend(validate_model_benign_test_coverage(manifest))

    join = descriptors.merge(
        manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="left",
    )
    summary = {}
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        test_ben = join[(join.vehicle_model == vm) & (join.split == "test") & join.attack_type.map(is_benign_attack_type)]
        summary[vm] = {
            "test_benign_descriptors": int(len(test_ben)),
            "test_benign_files": int(test_ben.source_file.nunique()) if not test_ben.empty else 0,
        }
    summary["validation_errors"] = errors
    summary["split_changed_from_original"] = True
    return manifest, summary
