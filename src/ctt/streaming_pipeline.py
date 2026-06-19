"""Single-pass streaming: normalize, window, and extract features per file."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import (
    MIN_VALID_FRAMES,
    NORMALIZED_COLUMNS,
    OUTPUT_ROOT,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from src.ctt.features import extract_window_features
from src.ctt.utils import (
    ensure_dir,
    parse_attack_from_filename,
    parse_data_field,
    vehicle_for_subset,
    write_markdown,
)
from src.ctt.windowing import window_label


def _process_file_args(args: tuple) -> tuple[list[dict], list[dict], dict]:
    """Worker entry point for parallel processing."""
    source_path, dataset_set, subset_name, output_root_str, window_size, stride = args
    return (*process_source_file_streaming(
        Path(source_path), dataset_set, subset_name, Path(output_root_str), window_size, stride
    ), {
        "source_file": str(source_path),
        "dataset_set": dataset_set,
        "subset_name": subset_name,
    })


def process_source_file_streaming(
    source_path: Path,
    dataset_set: str,
    subset_name: str,
    output_root: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[dict], list[dict]]:
    """Normalize, window, and extract features in one streaming pass."""
    _, attack_type, _ = parse_attack_from_filename(source_path.name)
    vehicle_id, manufacturer = vehicle_for_subset(dataset_set, subset_name)

    norm_dir = output_root / "normalized" / dataset_set / subset_name
    norm_dir.mkdir(parents=True, exist_ok=True)
    out_path = norm_dir / f"{source_path.stem}.csv"

    window_meta: list[dict] = []
    feature_rows: list[dict] = []
    frame_buffer: list[dict] = []
    source_row_index = 0
    global_frame_offset = 0  # frames already emitted from buffer
    window_id_base = abs(hash(str(source_path))) % 10_000_000
    window_counter = 0
    first_write = True
    benign_frames_for_profile: list[dict] = []

    chunksize = 500_000
    for chunk in pd.read_csv(source_path, chunksize=chunksize):
        out_rows: list[dict] = []
        for _, row in chunk.iterrows():
            dlc, byte_vals = parse_data_field(row.get("data_field", ""))
            can_id = str(row.get("arbitration_id", "")).strip().upper().replace("0X", "")
            attack_val = int(row.get("attack", 0))
            frame = {
                "timestamp": float(row.get("timestamp", np.nan)),
                "can_id": can_id,
                "dlc": dlc,
                "byte_0": byte_vals[0],
                "byte_1": byte_vals[1],
                "byte_2": byte_vals[2],
                "byte_3": byte_vals[3],
                "byte_4": byte_vals[4],
                "byte_5": byte_vals[5],
                "byte_6": byte_vals[6],
                "byte_7": byte_vals[7],
                "label": attack_val,
                "is_attack": attack_val,
                "attack_type": attack_type,
                "vehicle_id": vehicle_id,
                "manufacturer": manufacturer,
                "dataset_set": dataset_set,
                "subset_name": subset_name,
                "source_file": str(source_path),
                "source_row_index": source_row_index,
            }
            out_rows.append(frame)
            frame_buffer.append(frame)
            if attack_val == 0 and len(benign_frames_for_profile) < 5000:
                benign_frames_for_profile.append(frame)
            source_row_index += 1

            # Emit windows when buffer is large enough
            while len(frame_buffer) >= window_size:
                win_frames = frame_buffer[:window_size]
                start_idx = global_frame_offset
                label, purity = window_label(pd.Series([f["label"] for f in win_frames]))
                valid = sum(1 for f in win_frames if f["can_id"])
                if valid >= MIN_VALID_FRAMES:
                    wid = window_id_base * 1000 + window_counter
                    window_counter += 1
                    meta = {
                        "window_id": wid,
                        "normalized_path": str(out_path),
                        "start_frame_idx": start_idx,
                        "end_frame_idx": start_idx + window_size,
                        "window_size": window_size,
                        "n_frames": window_size,
                        "valid_frames": valid,
                        "label": label,
                        "label_purity": purity,
                        "is_attack": int(label) if not np.isnan(label) else 0,
                        "attack_type": attack_type,
                        "vehicle_id": vehicle_id,
                        "manufacturer": manufacturer,
                        "dataset_set": dataset_set,
                        "subset_name": subset_name,
                        "source_file": str(source_path),
                    }
                    window_meta.append(meta)
                    win_df = pd.DataFrame(win_frames)
                    feat = extract_window_features(win_df)
                    feature_rows.append({**meta, **feat})
                frame_buffer = frame_buffer[stride:]
                global_frame_offset += stride

        part_df = pd.DataFrame(out_rows, columns=NORMALIZED_COLUMNS)
        part_df.to_csv(out_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

    return window_meta, feature_rows


def run_streaming_pipeline(
    dataset_root: Path,
    output_root: Path = OUTPUT_ROOT,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run combined normalization, windowing, and feature extraction."""
    from src.ctt.utils import discover_ctt_files

    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")

    records = discover_ctt_files(dataset_root)
    norm_manifest: list[dict] = []
    all_window_meta: list[dict] = []
    all_features: list[dict] = []

    tasks = [
        (rec["source_file"], rec["dataset_set"], rec["subset_name"], str(output_root), WINDOW_SIZE, WINDOW_STRIDE)
        for rec in records
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_file_args, t): t for t in tasks}
        for future in as_completed(futures):
            win_meta, feats, info = future.result()
            all_window_meta.extend(win_meta)
            all_features.extend(feats)
            path = Path(info["source_file"])
            norm_manifest.append(
                {
                    "source_file": str(path),
                    "normalized_path": str(
                        output_root / "normalized" / info["dataset_set"] / info["subset_name"] / f"{path.stem}.csv"
                    ),
                    "dataset_set": info["dataset_set"],
                    "subset_name": info["subset_name"],
                    "row_count": 0,
                    "window_count": len(win_meta),
                }
            )
            completed += 1
            if completed % 10 == 0:
                print(f"  Streamed {completed}/{len(records)} files ({len(all_window_meta):,} windows)...")

    norm_df = pd.DataFrame(norm_manifest)
    norm_df.to_csv(manifest_dir / "normalization_manifest.csv", index=False)

    window_df = pd.DataFrame(all_window_meta)
    window_df.to_csv(manifest_dir / "window_manifest.csv", index=False)

    features_df = pd.DataFrame(all_features)
    features_path = output_root / "windows" / "all_window_features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)

    sections = {
        "Summary": (
            f"- Files processed: {len(records)}\n"
            f"- Windows: {len(window_df):,}\n"
            f"- Feature rows: {len(features_df):,}\n"
            f"- Single-pass streaming (normalize + window + features)"
        ),
    }
    write_markdown(audit_dir / "normalization_report.md", "Normalization Report", sections)
    write_markdown(audit_dir / "windowing_report.md", "Windowing Report", sections)

    return norm_df, window_df, features_df
