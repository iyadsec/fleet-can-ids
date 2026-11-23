"""Trace-level split manifest from fresh window features only."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.data_splits import (
    build_split_manifest_balanced_benign,
    is_benign_attack_type,
    validate_model_benign_test_coverage,
)
from src.experiments.model_diversity_final.split import _apply_within_trace_benign_splits


def _trace_id(row: pd.Series) -> str:
    return f"{row.get('vehicle_model', '')}::{row.get('source_file', '')}"


def build_e2e_split_manifest(
    features: pd.DataFrame,
    window_meta: pd.DataFrame,
    output_path: Path,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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

    if not window_meta.empty:
        seg = window_meta.groupby(["window_id", "vehicle_model", "source_file"], dropna=False).agg(
            segment_start=("start_frame_idx", "min"),
            segment_end=("end_frame_idx", "max"),
            frame_count=("n_frames", "max"),
            window_start=("start_frame_idx", "min"),
            window_end=("end_frame_idx", "max"),
        ).reset_index()
        join_cols = ["window_id", "vehicle_model", "source_file"]
        window_manifest = window_manifest.merge(seg, on=join_cols, how="left")

    grp = window_manifest.groupby(["source_file", "vehicle_model"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (src, vm), g in grp:
        split = str(g["split"].iloc[0])
        atk = str(g["attack_type"].iloc[0]) if "attack_type" in g.columns else "unknown"
        mal = int(g["ground_truth_malicious"].iloc[0])
        trace = _trace_id(pd.Series({"vehicle_model": vm, "source_file": src}))
        rows.append(
            {
                "source_file": src,
                "source_trace": trace,
                "vehicle_model": vm,
                "attack_type": atk,
                "ground_truth_malicious": mal,
                "split": split,
                "segment_start": int(g["segment_start"].min()) if "segment_start" in g.columns else "",
                "segment_end": int(g["segment_end"].max()) if "segment_end" in g.columns else "",
                "row_count": int(len(g)),
                "window_count": int(g["window_id"].nunique()),
                "split_group_id": hashlib.sha256(trace.encode()).hexdigest()[:16],
                "validation_status": "pending",
            }
        )
    trace_df = pd.DataFrame(rows)

    errors: list[str] = []
    if window_manifest.groupby("window_id")["split"].nunique().max() > 1:
        errors.append("window_id appears in multiple splits")
    errors.extend(validate_model_benign_test_coverage(window_manifest))

    if "event_id" in window_manifest.columns:
        if window_manifest.groupby("event_id")["split"].nunique().max() > 1:
            errors.append("event_id appears in multiple splits")

    trace_groups = trace_df.groupby("source_trace")["split"].nunique()
    if (trace_groups > 1).any():
        errors.append("source_trace appears in multiple splits")

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

    passed = len(errors) == 0
    trace_df["validation_status"] = "pass" if passed else "fail"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_df.to_csv(output_path, index=False)
    window_manifest.to_csv(output_path.parent / "window_split_manifest.csv", index=False)

    return trace_df, window_manifest, {"errors": errors, "passed": passed, "trace_count": len(trace_df)}
