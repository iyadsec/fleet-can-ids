"""Trace-grouped train/validation/test splits for leakage prevention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NORMAL_ATTACK_TYPES = {"normal", "attack_free", "benign", "none", "no_attack"}


def is_benign_attack_type(attack_type: str) -> bool:
    return str(attack_type).strip().lower() in NORMAL_ATTACK_TYPES


def build_split_manifest(
    metadata: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_columns: tuple[str, ...] = ("source_file", "vehicle_model"),
    seed: int = 42,
    attack_column: str = "attack_type",
) -> pd.DataFrame:
    """
    Assign trace groups to train / validation / test.

    Stratifies so source files containing attacks are represented in the test split
    (required for scenario evaluation).
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    cols = [c for c in group_columns if c in metadata.columns]
    if not cols:
        raise ValueError(f"Group columns missing from metadata: {group_columns}")

    file_col = "source_file" if "source_file" in metadata.columns else cols[0]
    files = metadata[[file_col, attack_column]].drop_duplicates(subset=[file_col])
    files["has_attack"] = ~files[attack_column].map(is_benign_attack_type)

    rng = np.random.default_rng(seed)
    attack_files = files.loc[files["has_attack"], file_col].unique()
    benign_files = files.loc[~files["has_attack"], file_col].unique()
    rng.shuffle(attack_files)
    rng.shuffle(benign_files)

    def _assign(file_list: np.ndarray) -> dict[str, str]:
        labels: dict[str, str] = {}
        n = len(file_list)
        n_train = int(n * train_ratio)
        n_val = int(n * validation_ratio)
        for i, f in enumerate(file_list):
            if i < n_train:
                labels[str(f)] = "train"
            elif i < n_train + n_val:
                labels[str(f)] = "validation"
            else:
                labels[str(f)] = "test"
        return labels

    split_map = _assign(attack_files)
    split_map.update(_assign(benign_files))

    groups = metadata[cols].drop_duplicates().reset_index(drop=True)
    groups["split"] = groups[file_col].astype(str).map(split_map).fillna("test")

    merged = metadata.merge(groups[cols + ["split"]], on=cols, how="left")
    if "window_id" in merged.columns:
        merged["window_key"] = (
            merged["vehicle_model"].astype(str)
            + "::"
            + merged.get("source_file", "").astype(str)
            + "::"
            + merged["window_id"].astype(str)
        )
    return merged


def save_split_manifest(manifest: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def validate_no_split_leakage(manifest: pd.DataFrame, group_columns: tuple[str, ...]) -> list[str]:
    cols = [c for c in group_columns if c in manifest.columns]
    errors: list[str] = []
    grouped = manifest.groupby(cols)["split"].nunique()
    leaky = grouped[grouped > 1]
    if len(leaky):
        errors.append(f"{len(leaky)} trace groups appear in multiple splits")
    test_attacks = manifest[
        (manifest["split"] == "test") & ~manifest.get("attack_type", pd.Series(dtype=str)).map(is_benign_attack_type)
    ]
    if test_attacks.empty and "attack_type" in manifest.columns:
        errors.append("test split contains no attack windows — scenarios cannot be evaluated")
    return errors
