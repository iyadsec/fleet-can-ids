"""Single-pass streaming: normalize, window, and extract features per file."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

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


def _parse_bytes_vectorized(data_fields: list[str]) -> tuple[list[int], list[list[float]]]:
    """Vectorized hex payload parsing."""
    dlcs = []
    all_bytes: list[list[float]] = []
    for df in data_fields:
        dlc, bv = parse_data_field(df)
        dlcs.append(dlc)
        all_bytes.append(bv)
    return dlcs, all_bytes


def _normalize_pl(source_path: Path, attack_type: str, vehicle_id: str, manufacturer: str,
                  dataset_set: str, subset_name: str) -> pd.DataFrame:
    """Fast normalization with polars."""
    df = pl.read_csv(source_path)
    data_fields = df["data_field"].to_list()
    dlcs, byte_lists = _parse_bytes_vectorized(data_fields)
    can_ids = df["arbitration_id"].cast(pl.Utf8).str.to_uppercase().str.replace("0X", "").to_list()
    attacks = df["attack"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    rows = {
        "timestamp": timestamps,
        "can_id": can_ids,
        "dlc": dlcs,
        "label": attacks,
        "is_attack": attacks,
        "attack_type": [attack_type] * len(df),
        "vehicle_id": [vehicle_id] * len(df),
        "manufacturer": [manufacturer] * len(df),
        "dataset_set": [dataset_set] * len(df),
        "subset_name": [subset_name] * len(df),
        "source_file": [str(source_path)] * len(df),
        "source_row_index": list(range(len(df))),
    }
    for i in range(8):
        rows[f"byte_{i}"] = [bv[i] for bv in byte_lists]

    return pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)


def _windows_from_normalized(
    norm_df: pd.DataFrame,
    out_path: Path,
    window_size: int,
    stride: int,
    source_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Generate windows and features from normalized dataframe (numpy-sliced)."""
    window_meta: list[dict] = []
    feature_rows: list[dict] = []
    n = len(norm_df)
    if n < window_size:
        return window_meta, feature_rows

    window_id_base = abs(hash(str(source_path))) % 10_000_000
    meta_cols = ["attack_type", "vehicle_id", "manufacturer", "dataset_set", "subset_name", "source_file"]
    meta_static = {c: norm_df[c].iloc[0] for c in meta_cols}

    timestamps = norm_df["timestamp"].to_numpy(dtype=float)
    can_ids = norm_df["can_id"].astype(str).to_numpy()
    dlc = norm_df["dlc"].to_numpy(dtype=float)
    labels = norm_df["label"].to_numpy(dtype=float)
    bytes_mat = norm_df[[f"byte_{i}" for i in range(8)]].to_numpy(dtype=float)

    from src.ctt.features import extract_window_features_numpy

    for wi, start in enumerate(range(0, n - window_size + 1, stride)):
        end = start + window_size
        w_can = can_ids[start:end]
        if np.sum(w_can != "nan") < MIN_VALID_FRAMES and np.sum(pd.notna(w_can)) < MIN_VALID_FRAMES:
            continue
        wlabels = labels[start:end]
        label = float(pd.Series(wlabels).value_counts().idxmax())
        purity = float((wlabels == label).mean())
        wid = window_id_base * 1000 + wi
        meta = {
            "window_id": wid,
            "normalized_path": str(out_path),
            "start_frame_idx": start,
            "end_frame_idx": end,
            "window_size": window_size,
            "n_frames": window_size,
            "valid_frames": int(np.sum((w_can != "") & (w_can != "nan"))),
            "label": label,
            "label_purity": purity,
            "is_attack": int(label),
            **meta_static,
        }
        window_meta.append(meta)
        feat = extract_window_features_numpy(
            timestamps[start:end], can_ids[start:end], dlc[start:end], bytes_mat[start:end]
        )
        feature_rows.append({**meta, **feat})

    return window_meta, feature_rows


def process_source_file_streaming(
    source_path: Path,
    dataset_set: str,
    subset_name: str,
    output_root: Path,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[dict], list[dict]]:
    """Normalize with polars, window, and extract features."""
    _, attack_type, _ = parse_attack_from_filename(source_path.name)
    vehicle_id, manufacturer = vehicle_for_subset(dataset_set, subset_name)

    norm_dir = output_root / "normalized" / dataset_set / subset_name
    norm_dir.mkdir(parents=True, exist_ok=True)
    out_path = norm_dir / f"{source_path.stem}.csv"

    if out_path.exists() and out_path.stat().st_size > 0:
        norm_df = pd.read_csv(out_path)
    else:
        norm_df = _normalize_pl(source_path, attack_type, vehicle_id, manufacturer, dataset_set, subset_name)
        norm_df.to_csv(out_path, index=False)

    # Attack-only train files are not used for benign onboarding or threshold calibration
    if subset_name == "train_01" and attack_type != "benign":
        return [], []

    return _windows_from_normalized(norm_df, out_path, window_size, stride, source_path)


def shard_path_for_file(features_dir: Path, dataset_set: str, subset_name: str, stem: str) -> Path:
    """Unique shard path per set/subset/file (avoids cross-set filename collisions)."""
    safe_subset = subset_name.replace("/", "_")
    return features_dir / f"features_{dataset_set}_{safe_subset}_{stem}.parquet"


def _process_file_args(args: tuple) -> tuple[list[dict], list[dict], dict]:
    source_path, dataset_set, subset_name, output_root_str, window_size, stride = args
    win_meta, feats = process_source_file_streaming(
        Path(source_path), dataset_set, subset_name, Path(output_root_str), window_size, stride
    )
    return win_meta, feats, {
        "source_file": str(source_path),
        "dataset_set": dataset_set,
        "subset_name": subset_name,
    }


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
    window_manifest_path = manifest_dir / "window_manifest.csv"
    window_header_written = window_manifest_path.exists() and window_manifest_path.stat().st_size > 0

    features_dir = ensure_dir(output_root / "windows" / "feature_shards")
    shard_paths: list[Path] = []
    tasks = []
    resumed_windows = 0

    for rec in records:
        path = Path(rec["source_file"])
        shard_path = shard_path_for_file(features_dir, rec["dataset_set"], rec["subset_name"], path.stem)
        out_path = output_root / "normalized" / rec["dataset_set"] / rec["subset_name"] / f"{path.stem}.csv"
        if shard_path.exists():
            shard_paths.append(shard_path)
            resumed_windows += 1
            norm_manifest.append({"source_file": str(path), "normalized_path": str(out_path),
                                   "dataset_set": rec["dataset_set"], "subset_name": rec["subset_name"],
                                   "window_count": 0, "resumed": True})
            continue
        # Skip attack-only train files (normalized but not windowed per benign-only protocol)
        if rec["subset_name"] == "train_01" and rec["attack_type"] != "benign":
            norm_manifest.append({"source_file": str(path), "normalized_path": str(out_path),
                                   "dataset_set": rec["dataset_set"], "subset_name": rec["subset_name"],
                                   "window_count": 0, "skipped_attack_train": True})
            continue
        tasks.append((rec["source_file"], rec["dataset_set"], rec["subset_name"], str(output_root), WINDOW_SIZE, WINDOW_STRIDE))

    completed = len(shard_paths)

    def _append_window_meta(rows: list[dict]) -> None:
        nonlocal window_header_written
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(window_manifest_path, mode="a", header=not window_header_written, index=False)
        window_header_written = True

    total_windows = 0

    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(_process_file_args, t): t for t in tasks}
        for future in as_completed(futures):
            win_meta, feats, info = future.result()
            _append_window_meta(win_meta)
            total_windows += len(win_meta)
            path = Path(info["source_file"])
            if feats:
                shard_path = shard_path_for_file(features_dir, info["dataset_set"], info["subset_name"], path.stem)
                pd.DataFrame(feats).to_parquet(shard_path, index=False)
                shard_paths.append(shard_path)
                del feats
            norm_manifest.append(
                {
                    "source_file": str(path),
                    "normalized_path": str(
                        output_root / "normalized" / info["dataset_set"] / info["subset_name"] / f"{path.stem}.csv"
                    ),
                    "dataset_set": info["dataset_set"],
                    "subset_name": info["subset_name"],
                    "window_count": len(win_meta),
                }
            )
            completed += 1
            if completed % 5 == 0:
                print(f"  Streamed {completed}/{len(records)} files ({total_windows:,} new windows)...", flush=True)

    norm_df = pd.DataFrame(norm_manifest)
    norm_df.to_csv(manifest_dir / "normalization_manifest.csv", index=False)

    # Build window manifest from shards if not complete
    if not window_manifest_path.exists() or window_manifest_path.stat().st_size == 0:
        meta_cols = ["window_id", "normalized_path", "start_frame_idx", "end_frame_idx", "window_size",
                       "n_frames", "valid_frames", "label", "label_purity", "is_attack", "attack_type",
                       "vehicle_id", "manufacturer", "dataset_set", "subset_name", "source_file"]
        for sp in shard_paths:
            feat_df = pd.read_parquet(sp)
            cols = [c for c in meta_cols if c in feat_df.columns]
            feat_df[cols].to_csv(window_manifest_path, mode="a", header=not window_header_written, index=False)
            window_header_written = True

    window_df = pd.read_csv(window_manifest_path) if window_manifest_path.exists() else pd.DataFrame()

    features_path = output_root / "windows" / "all_window_features.parquet"
    features_index_path = output_root / "windows" / "feature_shard_index.csv"
    pd.DataFrame({"shard_path": [str(p) for p in shard_paths]}).to_csv(features_index_path, index=False)

    # Build combined parquet only if small enough; otherwise use shards
    if shard_paths and len(shard_paths) <= 50:
        batches = []
        for i in range(0, len(shard_paths), 20):
            batch = pd.concat([pd.read_parquet(p) for p in shard_paths[i:i+20]], ignore_index=True)
            batches.append(batch)
        features_df = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        features_df.to_parquet(features_path, index=False)
    else:
        features_df = pd.DataFrame()
        if shard_paths:
            features_df = pd.read_parquet(shard_paths[0]).head(0)

    sections = {
        "Summary": (
            f"- Files processed: {len(records)}\n"
            f"- Windows: {len(window_df):,}\n"
            f"- Feature rows: {len(features_df):,}\n"
            f"- Polars-accelerated single-pass streaming"
        ),
    }
    write_markdown(audit_dir / "normalization_report.md", "Normalization Report", sections)
    write_markdown(audit_dir / "windowing_report.md", "Windowing Report", sections)

    return norm_df, window_df, features_df
