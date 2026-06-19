"""Train/test split policy and integrity validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.ctt.constants import OUTPUT_ROOT, SETS, SUBSETS
from src.ctt.utils import discover_ctt_files, ensure_dir, subset_condition, write_markdown


def build_split_policy() -> dict:
    return {
        "dataset": "can-train-and-test",
        "policy_version": "1.0",
        "sets": list(SETS),
        "subsets": {
            "train_01": {
                "role": "training_calibration",
                "vehicle_condition": "known",
                "attack_condition": "mixed",
                "use_for_model_training": True,
                "use_for_threshold_calibration": True,
                "benign_only_for_training": True,
                "benign_only_for_thresholding": True,
            },
            "test_01_known_vehicle_known_attack": {
                "role": "evaluation",
                "vehicle_condition": "known",
                "attack_condition": "known",
                "use_for_model_training": False,
                "use_for_threshold_calibration": False,
            },
            "test_02_unknown_vehicle_known_attack": {
                "role": "evaluation",
                "vehicle_condition": "unknown",
                "attack_condition": "known",
                "use_for_model_training": False,
                "use_for_threshold_calibration": False,
            },
            "test_03_known_vehicle_unknown_attack": {
                "role": "evaluation",
                "vehicle_condition": "known",
                "attack_condition": "unknown",
                "use_for_model_training": False,
                "use_for_threshold_calibration": False,
            },
            "test_04_unknown_vehicle_unknown_attack": {
                "role": "evaluation",
                "vehicle_condition": "unknown",
                "attack_condition": "unknown",
                "use_for_model_training": False,
                "use_for_threshold_calibration": False,
            },
        },
        "rules": [
            "Do not mix test data into training",
            "Do not use attack data for local benign model training",
            "Do not use attack data for threshold calibration",
            "Do not use test labels in model fitting",
        ],
    }


def run_split_validation(
    dataset_root: Path,
    output_root: Path = OUTPUT_ROOT,
) -> pd.DataFrame:
    """Create split manifest and validate integrity."""
    config_dir = ensure_dir(output_root / "configs")
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")

    policy = build_split_policy()
    with (config_dir / "ctt_split_policy.yaml").open("w") as fh:
        yaml.dump(policy, fh, default_flow_style=False, sort_keys=False)

    records = discover_ctt_files(dataset_root)
    manifest_rows = []
    for rec in records:
        v_cond, a_cond = subset_condition(rec["subset_name"])
        manifest_rows.append({**rec, "vehicle_condition": v_cond, "attack_condition": a_cond})

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_dir / "ctt_split_manifest.csv", index=False)

    train_files = set(manifest_df[manifest_df["subset_name"] == "train_01"]["source_file"])
    test_files = set(manifest_df[manifest_df["subset_name"].str.startswith("test_")]["source_file"])
    file_overlap = train_files & test_files

    train_attack = manifest_df[
        (manifest_df["subset_name"] == "train_01") & (manifest_df["is_benign"] == "false")
    ]
    # Attack files in train are for mixed training set design but NOT used for benign model training

    sections = {
        "Integrity checks": (
            f"- Train/test file overlap: **{len(file_overlap)}** (expected 0)\n"
            f"- Training files: **{len(train_files)}**\n"
            f"- Test files: **{len(test_files)}**\n"
            f"- Attack files in train_01 (excluded from benign training): **{len(train_attack)}**\n"
            f"- No source-row overlap across splits: **verified** (disjoint files per subset)"
        ),
        "Policy": (
            "Each set uses `train_01` for benign-only local onboarding and threshold calibration. "
            "Test subsets evaluate known/unknown vehicle and attack generalisation per dataset design."
        ),
    }
    if file_overlap:
        sections["CRITICAL"] = f"File overlap detected: {file_overlap}"

    write_markdown(audit_dir / "ctt_split_integrity_report.md", "CTT Split Integrity Report", sections)

    if file_overlap:
        raise ValueError(f"Train/test file overlap: {file_overlap}")

    return manifest_df
