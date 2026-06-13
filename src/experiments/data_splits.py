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


def validate_model_benign_test_coverage(
    manifest: pd.DataFrame,
    *,
    vehicle_models: tuple[str, ...] = ("Hyundai", "Kia", "Chevrolet"),
    attack_column: str = "attack_type",
) -> list[str]:
    """Return errors if any model lacks held-out test benign windows."""
    errors: list[str] = []
    if "vehicle_model" not in manifest.columns:
        return ["vehicle_model column missing from manifest"]
    test = manifest[manifest["split"] == "test"]
    for vm in vehicle_models:
        ben = test[
            (test["vehicle_model"] == vm)
            & test[attack_column].map(is_benign_attack_type)
        ]
        if ben.empty:
            errors.append(f"No test-split benign windows for {vm}")
    return errors


def build_split_manifest_balanced_benign(
    metadata: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_columns: tuple[str, ...] = ("source_file", "vehicle_model"),
    seed: int = 42,
    attack_column: str = "attack_type",
    min_benign_test_files_per_model: int = 1,
) -> pd.DataFrame:
    """
    Trace-grouped split with per-model guarantee of held-out benign traces.

    The original global benign-file shuffle can leave Kia/Chevrolet attack_free
    traces entirely in train when each model has few benign files. This builder
    stratifies benign assignment by vehicle_model so every platform has test-split
    benign data for heterogeneous fleet experiments.
    """
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    cols = [c for c in group_columns if c in metadata.columns]
    file_col = "source_file" if "source_file" in metadata.columns else cols[0]
    vm_col = "vehicle_model" if "vehicle_model" in metadata.columns else None

    files = metadata[[file_col, attack_column] + ([vm_col] if vm_col else [])].drop_duplicates(subset=[file_col])
    files["is_benign"] = files[attack_column].map(is_benign_attack_type)
    rng = np.random.default_rng(seed)
    split_map: dict[str, str] = {}

    def _assign_list(file_list: list[str], *, reserve_test: int = 0) -> None:
        arr = np.array(file_list, dtype=object)
        rng.shuffle(arr)
        n = len(arr)
        if n == 0:
            return
        n_test = max(reserve_test, int(np.ceil(n * test_ratio))) if reserve_test else int(np.ceil(n * test_ratio))
        n_test = min(n_test, n)
        n_val = int(n * validation_ratio)
        n_train = n - n_test - n_val
        if n_train < 0:
            n_train = 0
            n_val = max(0, n - n_test)
        for i, f in enumerate(arr):
            if i < n_test:
                split_map[str(f)] = "test"
            elif i < n_test + n_val:
                split_map[str(f)] = "validation"
            else:
                split_map[str(f)] = "train"

    if vm_col:
        attack_df = files[~files["is_benign"]]
        for vm, grp in attack_df.groupby(vm_col):
            _assign_list(grp[file_col].astype(str).unique().tolist())
        benign_df = files[files["is_benign"]]
        for vm, grp in benign_df.groupby(vm_col):
            _assign_list(
                grp[file_col].astype(str).unique().tolist(),
                reserve_test=min_benign_test_files_per_model,
            )
    else:
        attack_files = files.loc[~files["is_benign"], file_col].astype(str).unique().tolist()
        benign_files = files.loc[files["is_benign"], file_col].astype(str).unique().tolist()
        _assign_list(attack_files)
        _assign_list(benign_files, reserve_test=min_benign_test_files_per_model)

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
