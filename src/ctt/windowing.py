"""Windowing for normalized CTT data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import MIN_VALID_FRAMES, OUTPUT_ROOT, WINDOW_SIZE, WINDOW_STRIDE
from src.ctt.utils import ensure_dir, write_markdown


def window_label(labels: pd.Series) -> tuple[float, float]:
    """Return (window_label, label_purity)."""
    vals = pd.to_numeric(labels, errors="coerce").dropna()
    if vals.empty:
        return np.nan, 0.0
    counts = vals.value_counts()
    majority = float(counts.idxmax())
    purity = float(counts.max() / len(vals))
    return majority, purity


def generate_windows_for_normalized_file(
    norm_path: Path,
    output_dir: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    min_valid: int = MIN_VALID_FRAMES,
) -> list[dict]:
    """Generate sliding windows for one normalized file."""
    df = pd.read_csv(norm_path)
    n = len(df)
    if n < window_size:
        return []

    meta_rows: list[dict] = []
    window_id_base = abs(hash(str(norm_path))) % 10_000_000

    for wi, start in enumerate(range(0, n - window_size + 1, stride)):
        end = start + window_size
        chunk = df.iloc[start:end]
        valid_frames = int(chunk["can_id"].notna().sum())
        if valid_frames < min_valid:
            continue
        label, purity = window_label(chunk["label"])
        wid = window_id_base * 1000 + wi
        meta_rows.append(
            {
                "window_id": wid,
                "normalized_path": str(norm_path),
                "start_frame_idx": start,
                "end_frame_idx": end,
                "window_size": window_size,
                "n_frames": window_size,
                "valid_frames": valid_frames,
                "label": label,
                "label_purity": purity,
                "is_attack": int(label) if not np.isnan(label) else 0,
                "attack_type": chunk["attack_type"].iloc[0],
                "vehicle_id": chunk["vehicle_id"].iloc[0],
                "manufacturer": chunk["manufacturer"].iloc[0],
                "dataset_set": chunk["dataset_set"].iloc[0],
                "subset_name": chunk["subset_name"].iloc[0],
                "source_file": chunk["source_file"].iloc[0],
            }
        )

    if not meta_rows:
        return []

    return meta_rows


def run_windowing(
    output_root: Path = OUTPUT_ROOT,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> pd.DataFrame:
    """Generate windows for all normalized files."""
    norm_dir = output_root / "normalized"
    win_dir = ensure_dir(output_root / "windows")
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")

    norm_files = sorted(norm_dir.rglob("*.csv"))
    all_meta: list[dict] = []
    dropped_short = 0

    for i, path in enumerate(norm_files):
        rows = generate_windows_for_normalized_file(path, win_dir, window_size, stride)
        if not rows:
            dropped_short += 1
        all_meta.extend(rows)
        if (i + 1) % 20 == 0:
            print(f"  Windowed {i + 1}/{len(norm_files)} files...")

    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(manifest_dir / "window_manifest.csv", index=False)

    n_benign = int((meta_df["label"] == 0).sum()) if not meta_df.empty else 0
    n_attack = int((meta_df["label"] == 1).sum()) if not meta_df.empty else 0

    sections = {
        "Policy": (
            f"- window_size = {window_size}\n"
            f"- stride = {stride}\n"
            f"- minimum valid frames = {MIN_VALID_FRAMES}"
        ),
        "Summary": (
            f"- Files processed: {len(norm_files)}\n"
            f"- Files with no windows (too short): {dropped_short}\n"
            f"- Total windows: {len(meta_df):,}\n"
            f"- Benign windows: {n_benign:,}\n"
            f"- Attack windows: {n_attack:,}"
        ),
        "By vehicle": meta_df.groupby("vehicle_id").size().to_string() if not meta_df.empty else "N/A",
        "By attack type": meta_df.groupby("attack_type").size().to_string() if not meta_df.empty else "N/A",
    }
    write_markdown(audit_dir / "windowing_report.md", "Windowing Report", sections)
    return meta_df
