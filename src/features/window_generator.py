"""Sliding-window metadata generation over CAN frame sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

WINDOW_METADATA_COLUMNS = [
    "window_id",
    "vehicle_model",
    "attack_type",
    "label",
    "source_file",
    "start_frame_idx",
    "end_frame_idx",
    "window_size",
    "n_frames",
]

DEFAULT_WINDOW_SIZE = 100
DEFAULT_OVERLAP = 50


def resolve_window_params(
    config: dict[str, Any] | None = None,
    *,
    window_size: int | None = None,
    overlap: int | None = None,
    stride: int | None = None,
) -> tuple[int, int, int]:
    """
    Resolve window size, overlap, and stride.

    Precedence: explicit arguments > config ``features`` block.
    If *overlap* is set, ``stride = window_size - overlap``.
    If only *stride* is set, ``overlap = window_size - stride``.
    """
    feat = (config or {}).get("features", {})

    size = int(window_size if window_size is not None else feat.get("window_size", DEFAULT_WINDOW_SIZE))
    if size < 1:
        raise ValueError(f"window_size must be >= 1, got {size}")

    ov = overlap if overlap is not None else feat.get("overlap")
    st = stride if stride is not None else feat.get("stride")

    if ov is not None and st is not None and int(size) - int(ov) != int(st):
        logger.warning(
            "Both overlap (%s) and stride (%s) provided; using overlap (stride=%s).",
            ov,
            st,
            size - int(ov),
        )
        st = None

    if ov is not None:
        overlap_frames = int(ov)
        stride_frames = size - overlap_frames
    elif st is not None:
        stride_frames = int(st)
        overlap_frames = size - stride_frames
    else:
        overlap_frames = int(feat.get("overlap", DEFAULT_OVERLAP))
        stride_frames = size - overlap_frames

    if overlap_frames < 0 or overlap_frames >= size:
        raise ValueError(f"overlap must be in [0, {size - 1}], got {overlap_frames}")
    if stride_frames < 1:
        raise ValueError(f"stride must be >= 1 (overlap too large), got {stride_frames}")

    return size, overlap_frames, stride_frames


def _window_label(frame_labels: pd.Series) -> float:
    """Derive a single window label from frame-level labels (majority, ties -> max)."""
    values = pd.to_numeric(frame_labels, errors="coerce").dropna()
    if values.empty:
        return np.nan
    counts = values.value_counts()
    return float(counts.idxmax())


def generate_windows_for_trace(
    trace: pd.DataFrame,
    *,
    window_size: int,
    stride: int,
    source_file: str,
    id_offset: int = 0,
) -> pd.DataFrame:
    """Build sliding-window metadata rows for one CAN trace (one source file)."""
    n = len(trace)
    if n < window_size:
        return pd.DataFrame(columns=WINDOW_METADATA_COLUMNS)

    vehicle = trace["vehicle_model"].iloc[0]
    attack = trace["attack_type"].iloc[0]

    rows: list[dict[str, Any]] = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = trace.iloc[start:end]
        rows.append(
            {
                "window_id": id_offset + len(rows),
                "vehicle_model": vehicle,
                "attack_type": attack,
                "label": _window_label(window["label"]),
                "source_file": source_file,
                "start_frame_idx": start,
                "end_frame_idx": end,
                "window_size": window_size,
                "n_frames": len(window),
            }
        )

    return pd.DataFrame(rows, columns=WINDOW_METADATA_COLUMNS)


def generate_windows(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    window_size: int | None = None,
    overlap: int | None = None,
    stride: int | None = None,
    group_col: str = "source_file",
) -> pd.DataFrame:
    """
    Generate sliding-window metadata over CAN traffic.

    Windows are created **within each trace** (grouped by *group_col*), preserving
    temporal order inside each source file.
    """
    required = {group_col, "vehicle_model", "attack_type", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {sorted(missing)}")

    size, overlap_frames, stride_frames = resolve_window_params(
        config, window_size=window_size, overlap=overlap, stride=stride
    )
    logger.info(
        "Window params: size=%d, overlap=%d, stride=%d",
        size,
        overlap_frames,
        stride_frames,
    )

    parts: list[pd.DataFrame] = []
    next_id = 0
    for source_file, trace in df.groupby(group_col, sort=False):
        trace = trace.reset_index(drop=True)
        part = generate_windows_for_trace(
            trace,
            window_size=size,
            stride=stride_frames,
            source_file=str(source_file),
            id_offset=next_id,
        )
        if not part.empty:
            next_id += len(part)
            parts.append(part)

    if not parts:
        logger.warning("No windows generated (traces shorter than window_size?).")
        return pd.DataFrame(columns=WINDOW_METADATA_COLUMNS)

    meta = pd.concat(parts, ignore_index=True)
    meta["window_id"] = np.arange(len(meta), dtype=np.int64)
    return meta


def load_can_frames(path: Path | str) -> pd.DataFrame:
    """Load merged clean CAN CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CAN data not found: {csv_path}")
    logger.info("Loading CAN frames from %s", csv_path)
    return pd.read_csv(csv_path)


def save_window_metadata(meta: pd.DataFrame, path: Path | str) -> Path:
    """Persist window metadata table to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out, index=False)
    logger.info("Saved %d windows to %s", len(meta), out)
    return out


def window_statistics(meta: pd.DataFrame) -> dict[str, Any]:
    """Summarize generated windows."""
    if meta.empty:
        return {
            "n_windows": 0,
            "n_vehicles": 0,
            "n_attack_types": 0,
            "n_labels": 0,
        }
    return {
        "n_windows": int(len(meta)),
        "n_vehicles": int(meta["vehicle_model"].nunique()),
        "n_attack_types": int(meta["attack_type"].nunique()),
        "n_labels": int(meta["label"].nunique(dropna=True)),
        "windows_by_vehicle": meta.groupby("vehicle_model").size().to_dict(),
        "windows_by_attack": meta.groupby("attack_type").size().to_dict(),
        "windows_by_label": meta.groupby("label", dropna=False).size().to_dict(),
    }


def print_window_statistics(meta: pd.DataFrame) -> None:
    """Print human-readable window summary."""
    stats = window_statistics(meta)
    print("\n=== Window Statistics ===")
    print(f"Total windows:     {stats['n_windows']:,}")
    print(f"Vehicle models:    {stats['n_vehicles']}")
    print(f"Attack types:      {stats['n_attack_types']}")
    print(f"Distinct labels:   {stats['n_labels']}")
    if stats.get("windows_by_vehicle"):
        print("\n  Windows by vehicle:")
        for name, count in sorted(stats["windows_by_vehicle"].items()):
            print(f"    {name}: {count:,}")
    if stats.get("windows_by_attack"):
        print("\n  Windows by attack type:")
        for name, count in sorted(stats["windows_by_attack"].items()):
            print(f"    {name}: {count:,}")
    print("=========================\n")
