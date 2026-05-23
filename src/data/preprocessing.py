"""Clean, align, and persist CAN traces for downstream feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ProjectPaths


def preprocess_trace(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """
    Apply dataset-agnostic cleaning steps (sort by time, drop duplicates, etc.).

    Writes to *output_path* or ``data/processed/`` when provided.
    """
    if df.empty:
        raise ValueError("Cannot preprocess an empty trace.")
    # Placeholder: sort and deduplicate when timestamp column exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)
    if output_path is not None:
        out = Path(output_path)
        if not out.is_absolute() and paths is not None:
            out = paths.processed_dir / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    return df
