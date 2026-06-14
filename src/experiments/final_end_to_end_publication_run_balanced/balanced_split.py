"""Balanced platform-aware split with segment-level Chevrolet handling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.data_splits import is_benign_attack_type

WINDOW_SIZE = 100
GUARD_FRAMES = 100
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
PLATFORMS = ("Hyundai", "Kia", "Chevrolet")


def _basename(source_file: str) -> str:
    return Path(str(source_file)).name


def _segment_id(vehicle_model: str, source_file: str, split: str, part: str = "") -> str:
    base = f"{vehicle_model}::{_basename(source_file)}"
    if part:
        return f"{base}::{part}"
    return f"{base}::{split}"


def _assign_trace_level(files: list[str], *, seed: int, reserve_test: int = 1) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    arr = np.array(sorted(set(files)), dtype=object)
    rng.shuffle(arr)
    n = len(arr)
    if n == 0:
        return {}
    n_test = max(reserve_test, int(round(n * TEST_RATIO)))
    n_test = min(n_test, n)
    n_val = max(1, int(round(n * VAL_RATIO))) if n >= 3 else (1 if n >= 2 else 0)
    n_val = min(n_val, n - n_test)
    n_train = n - n_test - n_val
    if n_train < 1:
        n_train = 1
        n_val = max(0, n - n_test - n_train)
    out: dict[str, str] = {}
    for i, f in enumerate(arr):
        if i < n_test:
            out[str(f)] = "test"
        elif i < n_test + n_val:
            out[str(f)] = "validation"
        else:
            out[str(f)] = "train"
    return out


def _split_trace_segments(
    windows: pd.DataFrame,
    *,
    vehicle_model: str,
    source_file: str,
    attack_type: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], int]:
    """Contiguous window-order split with guard gaps between train/val/test."""
    w = windows.sort_values("start_frame_idx").reset_index(drop=True)
    n = len(w)
    if n < 9:
        raise ValueError(f"Insufficient windows for segment split ({n}) on {source_file}")

    n_val = max(1, int(round(n * VAL_RATIO)))
    n_test = max(1, int(round(n * TEST_RATIO)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    train_chunk = w.iloc[:n_train]
    val_chunk = w.iloc[n_train : n_train + n_val]
    test_chunk = w.iloc[n_train + n_val :]

    train_end = int(train_chunk["end_frame_idx"].max())
    guard1_start = train_end
    guard1_end = train_end + GUARD_FRAMES
    val_start = guard1_end
    val_end = int(val_chunk["end_frame_idx"].max())
    guard2_start = val_end
    guard2_end = val_end + GUARD_FRAMES
    test_start = guard2_end

    segment_rows: list[dict[str, Any]] = []
    assigned: list[pd.DataFrame] = []
    overlap_dropped = 0

    specs = [
        ("train", 0, train_end, "", guard1_start, guard1_end),
        ("validation", val_start, val_end, "seg_val", guard2_start, guard2_end),
        ("test", test_start, int(w["end_frame_idx"].max()), "seg_test", "", ""),
    ]

    for split, seg_start, seg_end, part, g_start, g_end in specs:
        if seg_end <= seg_start:
            continue
        seg_id = _segment_id(vehicle_model, source_file, split, part or split)
        in_seg = w[(w["start_frame_idx"] >= seg_start) & (w["end_frame_idx"] <= seg_end)].copy()
        if g_start != "" and g_end != "":
            in_guard = w[
                (w["start_frame_idx"] < g_end) & (w["end_frame_idx"] > g_start)
            ]
            overlap_dropped += len(in_guard)
            in_seg = in_seg[~in_seg["window_id"].isin(in_guard["window_id"])]

        if in_seg.empty:
            continue
        in_seg = in_seg.copy()
        in_seg["split"] = split
        in_seg["segment_id"] = seg_id
        in_seg["segment_start"] = seg_start
        in_seg["segment_end"] = seg_end
        in_seg["guard_start"] = g_start if g_start != "" else np.nan
        in_seg["guard_end"] = g_end if g_end != "" else np.nan
        assigned.append(in_seg)
        segment_rows.append(
            {
                "source_file": source_file,
                "vehicle_model": vehicle_model,
                "attack_type": attack_type,
                "split": split,
                "segment_id": seg_id,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "guard_start": g_start if g_start != "" else "",
                "guard_end": g_end if g_end != "" else "",
                "row_count": "",
                "window_count": len(in_seg),
                "overlap_count": 0,
                "validation_status": "pending",
                "split_method": "contiguous_segment",
            }
        )

    if not assigned:
        raise ValueError(f"No windows assigned after segment split for {source_file}")

    out = pd.concat(assigned, ignore_index=True)
    dropped = set(w["window_id"]) - set(out["window_id"])
    overlap_dropped += len(dropped)
    return out, segment_rows, overlap_dropped


def build_balanced_split(
    window_manifest: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (balanced_window_manifest, split_manifest, platform_summary, errors)."""
    wm = window_manifest.copy()
    if "attack_type" not in wm.columns:
        raise ValueError("window_manifest missing attack_type")

    assigned_parts: list[pd.DataFrame] = []
    split_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for vm in PLATFORMS:
        vm_w = wm[wm["vehicle_model"] == vm]
        if vm_w.empty:
            errors.append(f"No windows for {vm}")
            continue

        benign_files = (
            vm_w[vm_w["attack_type"].map(is_benign_attack_type)]["source_file"].astype(str).unique().tolist()
        )
        attack_files = (
            vm_w[~vm_w["attack_type"].map(is_benign_attack_type)]["source_file"].astype(str).unique().tolist()
        )

        benign_split_map: dict[str, str] = {}
        if len(benign_files) >= 3:
            benign_split_map = _assign_trace_level(benign_files, seed=seed + hash(vm) % 997)
        elif len(benign_files) == 2:
            ordered = sorted(benign_files, key=lambda f: len(vm_w[vm_w.source_file == f]))
            benign_split_map[str(ordered[0])] = "train"
        elif len(benign_files) == 1:
            pass
        else:
            errors.append(f"No benign source files for {vm}")

        attack_split_map = _assign_trace_level(attack_files, seed=seed + 17 + hash(vm) % 997, reserve_test=1)

        for src in sorted(set(benign_files + attack_files)):
            src_w = vm_w[vm_w["source_file"].astype(str) == src].copy()
            atk = str(src_w["attack_type"].iloc[0])
            is_ben = is_benign_attack_type(atk)

            if is_ben and src in benign_split_map and len(benign_files) >= 3:
                split = benign_split_map[src]
                part = src_w.copy()
                part["split"] = split
                part["segment_id"] = _segment_id(vm, src, split, "full_trace")
                part["segment_start"] = int(part["start_frame_idx"].min())
                part["segment_end"] = int(part["end_frame_idx"].max())
                part["guard_start"] = np.nan
                part["guard_end"] = np.nan
                assigned_parts.append(part)
                split_rows.append(
                    {
                        "source_file": src,
                        "vehicle_model": vm,
                        "attack_type": atk,
                        "split": split,
                        "segment_id": part["segment_id"].iloc[0],
                        "segment_start": int(part["segment_start"].iloc[0]),
                        "segment_end": int(part["segment_end"].iloc[0]),
                        "guard_start": "",
                        "guard_end": "",
                        "row_count": "",
                        "window_count": len(part),
                        "overlap_count": 0,
                        "validation_status": "pending",
                        "split_method": "complete_trace",
                    }
                )
            elif is_ben and len(benign_files) == 2 and src in benign_split_map:
                split = benign_split_map[src]
                part = src_w.copy()
                part["split"] = split
                part["segment_id"] = _segment_id(vm, src, split, "full_trace")
                part["segment_start"] = int(part["start_frame_idx"].min())
                part["segment_end"] = int(part["end_frame_idx"].max())
                part["guard_start"] = np.nan
                part["guard_end"] = np.nan
                assigned_parts.append(part)
                split_rows.append(
                    {
                        "source_file": src,
                        "vehicle_model": vm,
                        "attack_type": atk,
                        "split": split,
                        "segment_id": part["segment_id"].iloc[0],
                        "segment_start": int(part["segment_start"].iloc[0]),
                        "segment_end": int(part["segment_end"].iloc[0]),
                        "guard_start": "",
                        "guard_end": "",
                        "row_count": "",
                        "window_count": len(part),
                        "overlap_count": 0,
                        "validation_status": "pending",
                        "split_method": "complete_trace",
                    }
                )
            elif is_ben:
                seg_df, seg_meta, dropped = _split_trace_segments(
                    src_w, vehicle_model=vm, source_file=src, attack_type=atk, seed=seed
                )
                assigned_parts.append(seg_df)
                for row in seg_meta:
                    row["overlap_count"] = dropped
                    split_rows.append(row)
            elif src in attack_split_map:
                split = attack_split_map[src]
                if len(src_w) >= 9 and vm == "Chevrolet":
                    seg_df, seg_meta, dropped = _split_trace_segments(
                        src_w, vehicle_model=vm, source_file=src, attack_type=atk, seed=seed + 3
                    )
                    assigned_parts.append(seg_df)
                    for row in seg_meta:
                        row["overlap_count"] = dropped
                        split_rows.append(row)
                else:
                    part = src_w.copy()
                    part["split"] = split
                    part["segment_id"] = _segment_id(vm, src, split, "full_trace")
                    part["segment_start"] = int(part["start_frame_idx"].min())
                    part["segment_end"] = int(part["end_frame_idx"].max())
                    part["guard_start"] = np.nan
                    part["guard_end"] = np.nan
                    assigned_parts.append(part)
                    split_rows.append(
                        {
                            "source_file": src,
                            "vehicle_model": vm,
                            "attack_type": atk,
                            "split": split,
                            "segment_id": part["segment_id"].iloc[0],
                            "segment_start": int(part["segment_start"].iloc[0]),
                            "segment_end": int(part["segment_end"].iloc[0]),
                            "guard_start": "",
                            "guard_end": "",
                            "row_count": "",
                            "window_count": len(part),
                            "overlap_count": 0,
                            "validation_status": "pending",
                            "split_method": "complete_trace",
                        }
                    )

    balanced_wm = pd.concat(assigned_parts, ignore_index=True)
    split_manifest = pd.DataFrame(split_rows)

    for vm in PLATFORMS:
        for sp in ("train", "validation", "test"):
            ben = balanced_wm[
                (balanced_wm.vehicle_model == vm)
                & (balanced_wm.split == sp)
                & balanced_wm["attack_type"].map(is_benign_attack_type)
            ]
            if ben.empty:
                errors.append(f"No benign {sp} windows for {vm}")
            seg_count = split_manifest[
                (split_manifest.vehicle_model == vm)
                & (split_manifest.split == sp)
                & (split_manifest.attack_type.map(is_benign_attack_type))
            ]["segment_id"].nunique()
            if seg_count == 0:
                errors.append(f"No benign {sp} segments/traces for {vm}")

    if balanced_wm.groupby("window_id")["split"].nunique().max() > 1:
        errors.append("window_id appears in multiple splits")

    dup_events = balanced_wm.groupby("window_id")["split"].nunique()
    if (dup_events > 1).any():
        errors.append("duplicate window split assignment")

    platform_summary = _platform_summary(balanced_wm, split_manifest)
    passed = len(errors) == 0
    split_manifest["validation_status"] = "pass" if passed else "fail"
    return balanced_wm, split_manifest, platform_summary, errors


def _platform_summary(wm: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vm in PLATFORMS:
        for sp in ("train", "validation", "test"):
            sub = wm[(wm.vehicle_model == vm) & (wm.split == sp)]
            ben = sub[sub["attack_type"].map(is_benign_attack_type)]
            mal = sub[~sub["attack_type"].map(is_benign_attack_type)]
            seg = split_manifest[(split_manifest.vehicle_model == vm) & (split_manifest.split == sp)]
            rows.append(
                {
                    "vehicle_model": vm,
                    "split": sp,
                    "source_trace_count": int(seg[seg["split_method"] == "complete_trace"]["source_file"].nunique()),
                    "source_segment_count": int(seg[seg["split_method"] == "contiguous_segment"]["segment_id"].nunique()),
                    "benign_window_count": len(ben),
                    "malicious_window_count": len(mal),
                    "total_window_count": len(sub),
                }
            )
    return pd.DataFrame(rows)


def validate_balanced_split(wm: pd.DataFrame, split_manifest: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for vm in PLATFORMS:
        for sp in ("train", "validation", "test"):
            if wm[(wm.vehicle_model == vm) & (wm.split == sp) & wm.attack_type.map(is_benign_attack_type)].empty:
                errors.append(f"CHECK FAIL: {vm} missing benign {sp} windows")

    for (src, vm), grp in wm.groupby(["source_file", "vehicle_model"]):
        if grp["split"].nunique() > 1:
            if not grp["segment_id"].nunique() >= 1:
                errors.append(f"Source {src} spans splits without segment IDs")

    for seg_id, grp in wm.groupby("segment_id"):
        if grp["split"].nunique() > 1:
            errors.append(f"Segment {seg_id} spans multiple splits")

    for _, row in wm.iterrows():
        ws, we = int(row["start_frame_idx"]), int(row["end_frame_idx"])
        gs, ge = row.get("guard_start"), row.get("guard_end")
        if pd.notna(gs) and pd.notna(ge) and gs != "" and ge != "":
            if ws < int(ge) and we > int(gs):
                errors.append(f"Window {row['window_id']} overlaps guard region")

    return errors
