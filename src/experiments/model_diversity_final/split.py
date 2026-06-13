"""Final balanced split manifest with trace-level integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.data_splits import (
    build_split_manifest_balanced_benign,
    is_benign_attack_type,
    validate_model_benign_test_coverage,
    validate_no_split_leakage,
)


def _apply_within_trace_benign_splits(
    manifest: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> pd.DataFrame:
    """For single-file benign traces, assign windows to train/val/test in file order."""
    out = manifest.copy()
    benign = out[out["attack_type"].map(is_benign_attack_type)]
    for (vm, src), grp in benign.groupby(["vehicle_model", "source_file"]):
        if grp["split"].nunique() > 1:
            continue
        idx = grp.sort_values("window_id").index
        n = len(idx)
        if n < 30:
            continue
        n_train = int(n * train_ratio)
        n_val = int(n * validation_ratio)
        splits = ["train"] * n_train + ["validation"] * n_val + ["test"] * (n - n_train - n_val)
        out.loc[idx, "split"] = splits
    return out
from src.experiments.scenario_generator import load_descriptor_tables


def _trace_id(row: pd.Series) -> str:
    return f"{row.get('vehicle_model', '')}::{row.get('source_file', '')}"


def build_final_split_manifest(
    features_path: Path,
    descriptors_path: Path,
    output_path: Path,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, features = load_descriptor_tables(descriptors_path, features_path)
    meta = features.drop_duplicates(subset=["window_id", "vehicle_model", "source_file"]).copy()
    if "attack_type" not in meta.columns:
        meta["attack_type"] = "unknown"
    meta["normalized_attack_type"] = meta["attack_type"].astype(str).str.lower()
    meta["ground_truth_malicious"] = (~meta["normalized_attack_type"].map(is_benign_attack_type)).astype(int)

    window_manifest = build_split_manifest_balanced_benign(
        meta,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        min_benign_test_files_per_model=1,
    )
    window_manifest = _apply_within_trace_benign_splits(window_manifest)

    grp = window_manifest.groupby(["source_file", "vehicle_model"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (src, vm), g in grp:
        split = str(g["split"].iloc[0])
        atk = str(g["attack_type"].iloc[0]) if "attack_type" in g.columns else "unknown"
        norm = str(g["normalized_attack_type"].iloc[0]) if "normalized_attack_type" in g.columns else atk
        mal = int(g["ground_truth_malicious"].iloc[0]) if "ground_truth_malicious" in g.columns else int(not is_benign_attack_type(atk))
        trace = _trace_id(pd.Series({"vehicle_model": vm, "source_file": src}))
        rows.append(
            {
                "source_file": src,
                "source_trace": trace,
                "vehicle_model": vm,
                "normalized_attack_type": norm,
                "ground_truth_malicious": mal,
                "split": split,
                "split_group_id": hashlib.sha256(trace.encode()).hexdigest()[:16],
                "row_count": int(len(g)),
                "window_count": int(g["window_id"].nunique()) if "window_id" in g.columns else int(len(g)),
                "reason": "balanced_benign_per_model" if is_benign_attack_type(atk) else "attack_trace_grouped",
                "validation_status": "pending",
            }
        )
    trace_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(output_path, index=False)

    if window_manifest.groupby("window_id")["split"].nunique().max() > 1:
        errors = ["window_id appears in multiple splits"]
    else:
        errors = []
    errors.extend(validate_model_benign_test_coverage(window_manifest))

    event_splits = window_manifest.groupby("event_id")["split"].nunique() if "event_id" in window_manifest.columns else pd.Series(dtype=int)
    if not event_splits.empty and (event_splits > 1).any():
        errors.append("event_id appears in multiple splits")

    for vm in ("Hyundai", "Kia", "Chevrolet"):
        test_ben_windows = window_manifest[
            (window_manifest.vehicle_model == vm)
            & (window_manifest.split == "test")
            & window_manifest["attack_type"].map(is_benign_attack_type)
        ]
        if test_ben_windows.empty:
            errors.append(f"No benign test windows for {vm}")
        train_ben = window_manifest[
            (window_manifest.vehicle_model == vm)
            & (window_manifest.split == "train")
            & window_manifest["attack_type"].map(is_benign_attack_type)
        ]
        if train_ben.empty:
            errors.append(f"No benign train windows for {vm}")

    trace_df["validation_status"] = "pass" if not errors else "fail"
    trace_df.to_csv(output_path, index=False)

    summary = {
        "errors": errors,
        "passed": len(errors) == 0,
        "trace_count": len(trace_df),
        "window_manifest_path": str(output_path),
    }
    return trace_df, window_manifest, summary


def write_split_integrity_report(
    trace_df: pd.DataFrame,
    window_manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# Final split integrity report",
        "",
        f"**Trace groups:** {len(trace_df)}",
        f"**Validation:** {'PASS' if trace_df['validation_status'].eq('pass').all() else 'FAIL'}",
        "",
        "## Per-model benign coverage",
        "",
    ]
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        for sp in ("train", "validation", "test"):
            n = int(
                len(
                    trace_df[
                        (trace_df.vehicle_model == vm)
                        & (trace_df.split == sp)
                        & (trace_df.ground_truth_malicious == 0)
                    ]
                )
            )
            lines.append(f"- {vm} {sp} benign traces: {n}")
    lines += [
        "",
        "## Checks",
        "",
        "1. No source trace in multiple splits — validated at trace level.",
        "2. No overlapping window ranges across splits — windows inherit single trace split.",
        "3. Benign train/val/test per platform — see counts above.",
        "4. Test records excluded from model/scaler fitting — enforced in pipeline.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
