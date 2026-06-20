"""Single-pass streaming: normalize, window, and extract features per file."""

from __future__ import annotations

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
from src.ctt.progress_logger import ProgressLogger
from src.ctt.run_config import RunConfig
from src.ctt.utils import (
    ensure_dir,
    parse_attack_from_filename,
    parse_data_field,
    vehicle_for_subset,
    write_markdown,
)


def _parse_bytes_vectorized(data_fields: list[str]) -> tuple[list[int], list[list[float]]]:
    dlcs = []
    all_bytes: list[list[float]] = []
    for df in data_fields:
        dlc, bv = parse_data_field(df)
        dlcs.append(dlc)
        all_bytes.append(bv)
    return dlcs, all_bytes


def _normalize_pl(
    source_path: Path,
    attack_type: str,
    vehicle_id: str,
    manufacturer: str,
    dataset_set: str,
    subset_name: str,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Fast normalization with polars; optional row cap."""
    if max_rows is not None:
        df = pl.read_csv(source_path, n_rows=max_rows)
    else:
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
    max_windows: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Generate windows and features; stop at *max_windows* if set."""
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
        if max_windows is not None and len(window_meta) >= max_windows:
            break
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


def shard_path_for_file(features_dir: Path, dataset_set: str, subset_name: str, stem: str) -> Path:
    safe_subset = subset_name.replace("/", "_")
    return features_dir / f"features_{dataset_set}_{safe_subset}_{stem}.parquet"


def process_source_file(
    source_path: Path,
    dataset_set: str,
    subset_name: str,
    output_root: Path,
    *,
    max_rows_per_file: int | None = None,
    max_windows_for_file: int | None = None,
    skip_existing: bool = False,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[list[dict], list[dict], int]:
    """Normalize, window, extract features for one file. Returns (meta, feats, row_count)."""
    _, attack_type, _ = parse_attack_from_filename(source_path.name)
    vehicle_id, manufacturer = vehicle_for_subset(dataset_set, subset_name)

    norm_dir = output_root / "normalized" / dataset_set / subset_name
    norm_dir.mkdir(parents=True, exist_ok=True)
    out_path = norm_dir / f"{source_path.stem}.csv"

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        norm_df = pd.read_csv(out_path)
    else:
        norm_df = _normalize_pl(
            source_path, attack_type, vehicle_id, manufacturer,
            dataset_set, subset_name, max_rows=max_rows_per_file,
        )
        norm_df.to_csv(out_path, index=False)

    row_count = len(norm_df)

    if subset_name == "train_01" and attack_type != "benign":
        return [], [], row_count

    win_meta, feats = _windows_from_normalized(
        norm_df, out_path, window_size, stride, source_path,
        max_windows=max_windows_for_file,
    )
    return win_meta, feats, row_count


def _select_records(cfg: RunConfig, records: list[dict]) -> list[dict]:
    """Select files for processing; round-robin by subset for set_pilot."""
    from collections import defaultdict

    from src.ctt.constants import SUBSETS

    filtered = [r for r in records if cfg.should_process_record(r)]
    if cfg.stage != "set_pilot":
        selected: list[dict] = []
        for rec in filtered:
            selected.append(rec)
            if cfg.max_files is not None and len(selected) >= cfg.max_files:
                break
        return selected

    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in filtered:
        buckets[rec["subset_name"]].append(rec)
    subset_order = [s for s in SUBSETS if buckets.get(s)]
    selected = []
    while subset_order:
        if cfg.max_files is not None and len(selected) >= cfg.max_files:
            break
        progressed = False
        for subset in subset_order:
            if buckets[subset]:
                selected.append(buckets[subset].pop(0))
                progressed = True
                if cfg.max_files is not None and len(selected) >= cfg.max_files:
                    break
        if not progressed:
            break
    return selected


def _subset_window_budgets(cfg: RunConfig, selected: list[dict]) -> dict[str, int]:
    """Allocate global window cap evenly across subsets (set_pilot)."""
    if cfg.stage != "set_pilot" or cfg.max_windows is None:
        return {}
    subsets = sorted({r["subset_name"] for r in selected})
    if not subsets:
        return {}
    per_subset = max(cfg.max_windows // len(subsets), 1)
    return {s: per_subset for s in subsets}


def run_streaming_pipeline(
    dataset_root: Path,
    output_root: Path = OUTPUT_ROOT,
    config: RunConfig | None = None,
    progress: ProgressLogger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run normalization, windowing, and feature extraction with optional caps."""
    from src.ctt.utils import discover_ctt_files

    cfg = config or RunConfig(stage="full", dataset_root=dataset_root, output_root=output_root)
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")
    features_dir = ensure_dir(output_root / "windows" / "feature_shards")

    stage_suffix = "" if cfg.stage in ("full", "set_pilot") else f"_{cfg.stage}"
    window_manifest_path = manifest_dir / f"window_manifest{stage_suffix}.csv"
    window_header_written = (
        cfg.resume and window_manifest_path.exists() and window_manifest_path.stat().st_size > 0
    )

    records = discover_ctt_files(dataset_root)
    selected = _select_records(cfg, records)
    subset_budgets = _subset_window_budgets(cfg, selected)
    subset_windows_used: dict[str, int] = {s: 0 for s in subset_budgets}

    norm_manifest: list[dict] = []
    shard_paths: list[Path] = []
    all_feature_rows: list[dict] = []
    total_windows = 0
    global_window_cap = cfg.max_windows

    if cfg.resume and window_manifest_path.exists() and window_manifest_path.stat().st_size > 0:
        existing_windows = pd.read_csv(window_manifest_path)
        total_windows = len(existing_windows)
        if cfg.stage == "set_pilot" and subset_budgets and not existing_windows.empty:
            counts = existing_windows.groupby("subset_name").size().to_dict()
            for subset, cap in subset_budgets.items():
                subset_windows_used[subset] = int(counts.get(subset, 0))

    if progress:
        progress.info(
            f"Processing {len(selected)} files (stage={cfg.stage}, "
            f"max_rows={cfg.max_rows_per_file}, max_windows={global_window_cap}, "
            f"subset_budgets={subset_budgets or 'none'})"
        )
        if cfg.resume and total_windows > 0:
            progress.info(f"Resuming with {total_windows:,} existing windows")

    processed_sources: set[str] = set()
    norm_manifest_path = manifest_dir / f"normalization_manifest{stage_suffix}.csv"
    if cfg.resume and norm_manifest_path.exists():
        processed_sources = set(pd.read_csv(norm_manifest_path)["source_file"].astype(str))

    shard_index_path = output_root / "windows" / f"feature_shard_index{stage_suffix}.csv"
    if cfg.resume and shard_index_path.exists():
        shard_paths = [Path(p) for p in pd.read_csv(shard_index_path)["shard_path"].tolist()]

    for rec in selected:
        if global_window_cap is not None and total_windows >= global_window_cap:
            if progress:
                progress.info(f"Global window cap reached ({global_window_cap:,}); stopping.")
            break

        path = Path(rec["source_file"])
        subset = rec["subset_name"]
        if subset in subset_budgets and subset_windows_used.get(subset, 0) >= subset_budgets[subset]:
            continue

        shard_path = shard_path_for_file(features_dir, rec["dataset_set"], rec["subset_name"], path.stem)

        if cfg.skip_existing and shard_path.exists() and str(path) in processed_sources:
            if shard_path not in shard_paths:
                shard_paths.append(shard_path)
            if progress:
                progress.file_processed(str(path), rows=0, windows=0)
            continue

        remaining_global = global_window_cap - total_windows if global_window_cap is not None else None
        remaining_subset = None
        if subset in subset_budgets:
            remaining_subset = subset_budgets[subset] - subset_windows_used.get(subset, 0)
        if remaining_global is not None and remaining_subset is not None:
            remaining = min(remaining_global, remaining_subset)
        else:
            remaining = remaining_global if remaining_global is not None else remaining_subset
        if remaining is not None and remaining <= 0:
            continue

        win_meta, feats, row_count = process_source_file(
            path, rec["dataset_set"], rec["subset_name"], output_root,
            max_rows_per_file=cfg.max_rows_per_file,
            max_windows_for_file=remaining,
            skip_existing=cfg.skip_existing,
        )

        if progress:
            progress.file_processed(str(path), rows=row_count, windows=len(win_meta))

        if win_meta:
            meta_df = pd.DataFrame(win_meta)
            meta_df.to_csv(window_manifest_path, mode="a", header=not window_header_written, index=False)
            window_header_written = True

        if feats:
            pd.DataFrame(feats).to_parquet(shard_path, index=False)
            shard_paths.append(shard_path)
            all_feature_rows.extend(feats)

        total_windows += len(win_meta)
        if subset in subset_windows_used:
            subset_windows_used[subset] += len(win_meta)
        norm_manifest.append({
            "source_file": str(path),
            "normalized_path": str(
                output_root / "normalized" / rec["dataset_set"] / rec["subset_name"] / f"{path.stem}.csv"
            ),
            "dataset_set": rec["dataset_set"],
            "subset_name": rec["subset_name"],
            "row_count": row_count,
            "window_count": len(win_meta),
        })

    if cfg.resume and norm_manifest_path.exists():
        prior = pd.read_csv(norm_manifest_path)
        new_df = pd.DataFrame(norm_manifest)
        if not new_df.empty:
            combined = pd.concat([prior, new_df], ignore_index=True).drop_duplicates(
                subset=["source_file"], keep="last"
            )
        else:
            combined = prior
        combined.to_csv(norm_manifest_path, index=False)
    else:
        pd.DataFrame(norm_manifest).to_csv(norm_manifest_path, index=False)

    window_df = pd.read_csv(window_manifest_path) if window_manifest_path.exists() else pd.DataFrame()
    features_path = output_root / "windows" / f"all_window_features{stage_suffix}.parquet"
    pd.DataFrame({"shard_path": [str(p) for p in shard_paths]}).to_csv(
        output_root / "windows" / f"feature_shard_index{stage_suffix}.csv", index=False
    )

    if all_feature_rows:
        features_df = pd.DataFrame(all_feature_rows)
        features_df.to_parquet(features_path, index=False)
    elif shard_paths:
        features_df = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
        features_df.to_parquet(features_path, index=False)
    else:
        features_df = pd.DataFrame()

    sections = {
        "Summary": (
            f"- Stage: {cfg.stage}\n"
            f"- Files processed: {len(norm_manifest)}\n"
            f"- Windows: {len(window_df):,}\n"
            f"- Feature rows: {len(features_df):,}\n"
            f"- max_rows_per_file: {cfg.max_rows_per_file}\n"
            f"- max_windows: {cfg.max_windows}"
        ),
    }
    write_markdown(audit_dir / f"normalization_report{stage_suffix}.md", "Normalization Report", sections)
    write_markdown(audit_dir / f"windowing_report{stage_suffix}.md", "Windowing Report", sections)

    return pd.DataFrame(norm_manifest), window_df, features_df
