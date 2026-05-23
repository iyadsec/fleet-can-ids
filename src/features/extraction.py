"""Sliding-window and statistical features for CAN intrusion detection."""

from __future__ import annotations

from typing import Any

import pandas as pd


def extract_window_features(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build per-window feature rows from a preprocessed trace.

    Uses ``features.window_size`` and ``features.stride`` from *config*.
    Full feature set (ID histograms, inter-arrival times, payload stats) TBD.
    """
    feat_cfg = config.get("features", {})
    window_size = int(feat_cfg.get("window_size", 100))
    stride = int(feat_cfg.get("stride", 50))
    if len(df) < window_size:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start : start + window_size]
        rows.append(
            {
                "window_start": start,
                "window_end": start + window_size,
                "n_frames": len(window),
            }
        )
    return pd.DataFrame(rows)
