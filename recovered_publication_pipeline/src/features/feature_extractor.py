"""Behavioural feature extraction from CAN sliding windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

BYTE_COLS = [f"byte{i}" for i in range(8)]

BEHAVIOURAL_FEATURE_COLUMNS: list[str] = [
    "frame_count",
    "unique_can_id_count",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "mean_dlc",
    "std_dlc",
    *[f"byte_mean_{i}" for i in range(8)],
    *[f"byte_std_{i}" for i in range(8)],
]

METADATA_COLUMNS = [
    "window_id",
    "vehicle_model",
    "attack_type",
    "label",
    "source_file",
    "start_frame_idx",
    "end_frame_idx",
]


def _can_id_entropy(can_ids: np.ndarray) -> float:
    if can_ids.size == 0:
        return np.nan
    _, counts = np.unique(can_ids, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    """Compute behavioural features for one window of CAN frames."""
    n = len(window)
    features: dict[str, float] = {"frame_count": float(n)}

    if n == 0:
        for col in BEHAVIOURAL_FEATURE_COLUMNS[1:]:
            features[col] = np.nan
        return features

    can_ids = window["can_id"].astype(str).to_numpy()
    features["unique_can_id_count"] = float(len(np.unique(can_ids)))
    features["can_id_entropy"] = _can_id_entropy(can_ids)

    _, counts = np.unique(can_ids, return_counts=True)
    features["most_common_can_id_ratio"] = float(counts.max() / n)

    timestamps = pd.to_numeric(window["timestamp"], errors="coerce").to_numpy()
    if n > 1:
        inter = np.diff(timestamps)
        inter = inter[np.isfinite(inter)]
        features["mean_inter_arrival_time"] = float(np.mean(inter)) if inter.size else np.nan
        features["std_inter_arrival_time"] = float(np.std(inter)) if inter.size else np.nan
    else:
        features["mean_inter_arrival_time"] = np.nan
        features["std_inter_arrival_time"] = np.nan

    dlc = pd.to_numeric(window["dlc"], errors="coerce").to_numpy()
    features["mean_dlc"] = float(np.nanmean(dlc))
    features["std_dlc"] = float(np.nanstd(dlc))

    for i, col in enumerate(BYTE_COLS):
        if col in window.columns:
            vals = pd.to_numeric(window[col], errors="coerce").to_numpy()
            features[f"byte_mean_{i}"] = float(np.nanmean(vals))
            features[f"byte_std_{i}"] = float(np.nanstd(vals))
        else:
            features[f"byte_mean_{i}"] = np.nan
            features[f"byte_std_{i}"] = np.nan

    return features


def extract_features_for_trace(
    trace: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Extract features for all windows belonging to one CAN trace."""
    rows: list[dict[str, Any]] = []
    for _, meta in windows.iterrows():
        start = int(meta["start_frame_idx"])
        end = int(meta["end_frame_idx"])
        chunk = trace.iloc[start:end]
        feat = extract_window_features(chunk)
        row = {col: meta[col] for col in METADATA_COLUMNS if col in meta}
        row.update(feat)
        rows.append(row)
    return pd.DataFrame(rows)


def extract_features(
    frames: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract behavioural features for every row in *windows*.

    Frames are joined to windows via ``source_file`` and frame indices.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore

    grouped_windows = list(windows.groupby("source_file", sort=False))
    iterator: Any = grouped_windows
    if show_progress and tqdm is not None:
        iterator = tqdm(grouped_windows, desc="Extracting window features")

    parts: list[pd.DataFrame] = []
    frames_by_source = {k: g.reset_index(drop=True) for k, g in frames.groupby("source_file", sort=False)}

    for source_file, win_group in iterator:
        trace = frames_by_source.get(source_file)
        if trace is None or trace.empty:
            logger.warning("No frames for source_file=%s", source_file)
            continue
        parts.append(extract_features_for_trace(trace, win_group))

    if not parts:
        return pd.DataFrame(columns=METADATA_COLUMNS + BEHAVIOURAL_FEATURE_COLUMNS)

    out = pd.concat(parts, ignore_index=True)
    if "window_id" in out.columns:
        out = out.sort_values("window_id").reset_index(drop=True)
    return out[METADATA_COLUMNS + BEHAVIOURAL_FEATURE_COLUMNS]


def load_frames(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_windows(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_window_features(features: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    logger.info("Saved %d feature rows to %s", len(features), out)
    return out


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in df.columns]


def plot_feature_correlation_heatmap(
    features: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: tuple[float, float] = (14, 12),
) -> Path:
    """Save Pearson correlation heatmap of behavioural features."""
    cols = _numeric_feature_columns(features)
    corr = features[cols].corr(numeric_only=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=False,
        linewidths=0.2,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Behavioural feature correlation")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote correlation heatmap to %s", out)
    return out


def plot_feature_distributions(
    features: pd.DataFrame,
    output_path: Path | str,
    *,
    cols: list[str] | None = None,
    sample_max: int | None = 20_000,
) -> Path:
    """Save histogram grid of behavioural feature distributions."""
    plot_cols = cols or _numeric_feature_columns(features)
    data = features[plot_cols]
    if sample_max is not None and len(data) > sample_max:
        data = data.sample(n=sample_max, random_state=42)

    n = len(plot_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes_flat, plot_cols):
        series = data[col].dropna()
        ax.hist(series, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(plot_cols) :]:
        ax.axis("off")

    fig.suptitle("Behavioural feature distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote feature distributions to %s", out)
    return out


def print_feature_summary(features: pd.DataFrame) -> None:
    """Print brief extraction summary."""
    cols = _numeric_feature_columns(features)
    print("\n=== Feature Extraction Summary ===")
    print(f"Windows:          {len(features):,}")
    print(f"Feature columns:  {len(cols)}")
    print(f"Missing values:   {int(features[cols].isna().sum().sum())}")
    if "vehicle_model" in features.columns:
        print("\n  By vehicle:")
        for name, count in features["vehicle_model"].value_counts().items():
            print(f"    {name}: {count:,}")
    print("==================================\n")
